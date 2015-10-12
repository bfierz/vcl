/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

// Abseil
#include <absl/container/inlined_vector.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/util/mortoncodes.h>

namespace Vcl { namespace Core
{
	class Octree
	{
		using MortonCodeIterator = std::vector<std::pair<uint64_t, int>>::const_iterator;

		//! Octree node
		//! This class is used for both inner nodes and leafs.
		struct Node
		{
			Node() = default;
			Node(const Eigen::AlignedBox3f& box)
			: Box{ box }
			{
				Children.assign(-1);
			}

			//! Check if this node is a leaf-node
			//! \returns True if the node is a leaf-node
			bool isLeaf() const { return Children[0] == -1; }

			//! Aligned bounding box of the node
			const Eigen::AlignedBox3f Box;
			
			//! Pointers to child octants
			std::array<int, 8> Children;

			//! Data stored in the node
			absl::InlinedVector<int, 8> Data;
		};

	public:
		Octree
		(
			const Eigen::AlignedBox3f& box,
			int max_depth
		)
		: _maxDepth(std::min(max_depth, 21))
		, _topNode(box)
		{
			VclRequire(max_depth <= 21, "Depth of maximum 21 levels is supported.");

			// Compute smallest cell size
			const int nr_cells = 1 << (_maxDepth - 1);
			_cellSize = (box.sizes()).array() /
				Eigen::Vector3f::Constant(nr_cells).array();

			// Start with an empty tree
			clear();
		}

	public:
		//! Clear the content of the tree
		void clear()
		{
			// Delete old tree
			_nodes.clear();

			// Allocate new top-level node
			_topNode.Children.assign(-1);
		}

		void assign(const std::vector<Eigen::Vector3f>& points)
		{
			// Clear the current tree
			clear();

			// Early out
			if (points.size() == 0)
				return;

			// Preallocate nodes
			_nodes.reserve(points.size());

			// Size of cell = full extent / number of cells
			const Eigen::Vector3f inv_cell_size =
				Eigen::Vector3f::Ones().array() / _cellSize.array();

			// Lower corner
			const Eigen::Vector3f lower = _topNode.Box.min();

			// Prepare the morton codes
			std::vector<std::pair<uint64_t, int>> codes;
			codes.reserve(points.size() + 1);
			for (int i = 0; i < points.size(); i++)
			{
				const Eigen::Vector3f p = (points[i] - lower).array() * inv_cell_size.array();
				codes.emplace_back(std::make_pair(Util::MortonCode::encode(p.x(), p.y(), p.z()), i));
			}
			// Insert decodeable end-token
			codes.emplace_back(std::make_pair(Util::MortonCode::encode(0x1fffff, 0x1fffff, 0x1fffff), -1));

			_points = points;

			std::sort(codes.begin(), codes.end(), [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b)
			{
				return a.first < b.first;
			});

			// Determine first split node
			uint64_t top = codes.back().first;
			int level = (msbIdx(top) / 3);

			// Start building the tree
			split(_topNode, codes.begin(), codes.end() - 1, level, points);
		}

		size_t find(Vector3f p, float radius, std::vector<int>& enclosed_points)
		{
			const Eigen::Vector3f inv_cell_size =
				Eigen::Vector3f::Constant(1).array() / _cellSize.array();

			// Search-box
			Eigen::AlignedBox3f search_box{ p - Eigen::Vector3f::Constant(radius), p + Eigen::Vector3f::Constant(radius) };
			search_box = _topNode.Box.intersection(search_box);

			const auto lb = search_box.min();
			const auto lb_code = Util::MortonCode::encode(lb.x(), lb.y(), lb.z());

			// Maximum point search-box
			const auto ub = search_box.max();
			const auto ub_code = Util::MortonCode::encode(ub.x(), ub.y(), ub.z());

			const auto validator = [center = p, r2 = radius* radius](const Eigen::Vector3f& p)
			{ return (p - center).squaredNorm() <= r2; };
			find(0, _topNode, search_box, validator, enclosed_points);

			return enclosed_points.size();
		}

		size_t find(const Eigen::AlignedBox3f& search_box, std::vector<int>& enclosed_points)
		{
			const auto validator = [search_box](const Eigen::Vector3f& p) { return search_box.contains(p); };
			find(0, _topNode, search_box, validator, enclosed_points);
			return enclosed_points.size();
		}

	private:
		void split
		(
			Node& n,
			MortonCodeIterator begin, MortonCodeIterator end,
			int level,
			const std::vector<Eigen::Vector3f>& points
		)
		{
			if (level == 0 || end - begin <= 8)
			{
				for (auto it = begin; it != end; ++it)
				{
					n.Data.emplace_back(it->second);
				}
				return;
			}

			auto splitIterZ = splitDim(2, begin, end, level);

			auto splitIterY0 = splitDim(1, begin, splitIterZ, level);
			auto splitIterY1 = splitDim(1, splitIterZ, end, level);

			auto splitIterX0 = splitDim(0, begin, splitIterY0, level);
			auto splitIterX1 = splitDim(0, splitIterY0, splitIterZ, level);

			auto splitIterX2 = splitDim(0, splitIterZ, splitIterY1, level);
			auto splitIterX3 = splitDim(0, splitIterY1, end, level);

			std::array<std::pair<MortonCodeIterator, MortonCodeIterator>, 8> octants;
			octants[0] = std::make_pair(begin, splitIterX0);
			octants[1] = std::make_pair(splitIterX0, splitIterY0);
			octants[2] = std::make_pair(splitIterY0, splitIterX1);
			octants[3] = std::make_pair(splitIterX1, splitIterZ);

			octants[4] = std::make_pair(splitIterZ, splitIterX2);
			octants[5] = std::make_pair(splitIterX2, splitIterY1);
			octants[6] = std::make_pair(splitIterY1, splitIterX3);
			octants[7] = std::make_pair(splitIterX3, end);

			const Eigen::Vector3f new_half_dim = 0.25f * n.Box.sizes();
			for (int i = 0; i < 8; i++)
			{
				Eigen::Vector3f new_origin = n.Box.center();
				new_origin.x() += new_half_dim.x() * (i & 1 ? 1.0f : -1.0f);
				new_origin.y() += new_half_dim.y() * (i & 2 ? 1.0f : -1.0f);
				new_origin.z() += new_half_dim.z() * (i & 4 ? 1.0f : -1.0f);

				Eigen::AlignedBox3f new_box
				{
					new_origin - new_half_dim, new_origin + new_half_dim
				};

				VclAssertBlock
				{
					for (auto it = octants[i].first; it != octants[i].second; ++it)
					{
						VclCheck(new_box.contains(points[it->second]), "Point is in octant");
					}
				}

				_nodes.emplace_back(new_box);
				auto& new_node = _nodes.back();
				n.Children[i] = _nodes.size() - 1;

				split(new_node, octants[i].first, octants[i].second, level - 1, points);
			}
		}

		MortonCodeIterator splitDim(int dim, MortonCodeIterator beg, MortonCodeIterator end, int level)
		{
			// Early out
			if (beg == end)
				return end;

			// Split value
			const uint64_t split_value = 1 << (level * 3 + dim);
			const uint64_t mask = ((split_value << 1) - 1);

			// Find the split indices
			const auto mid =
				std::lower_bound(beg, end, std::make_pair(split_value, -1),
					[mask](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b)
			{
				return (a.first & mask) < (b.first & mask);
			});

			VclAssertBlock
			{
				std::array<uint32_t, 3> coord_mid;
				Vcl::Util::MortonCode::decode(mid->first, coord_mid[0], coord_mid[1], coord_mid[2]);

				for (auto code_it = beg; code_it != mid; ++code_it)
				{
					std::array<uint32_t, 3> coord;
					Vcl::Util::MortonCode::decode(code_it->first, coord[0], coord[1], coord[2]);

					VclCheck(coord[dim] <= coord_mid[dim], "Coordinate is in range");
				}
				for (auto code_it = mid; code_it != end; ++code_it)
				{
					std::array<uint32_t, 3> coord;
					Vcl::Util::MortonCode::decode(code_it->first, coord[0], coord[1], coord[2]);

					VclCheck(coord_mid[dim] <= coord[dim], "Coordinate is in range");
				}
			}

			return mid;
		}

		template<typename Func>
		void find(int node_id, const Node& node, const Eigen::AlignedBox3f& search_box, const Func& validator, std::vector<int>& enclosed_points)
		{
			if (node.isLeaf())
			{
				if (!search_box.intersects(node.Box))
					return;

				// Collect data
				const auto& points = _points;
				std::copy_if(node.Data.begin(), node.Data.end(),
					std::back_inserter(enclosed_points), [&validator, &points](int idx)
				{
					return validator(points[idx]);
				});
			}
			else
			{
				for (int i = 0; i < 8; ++i)
				{
					VclCheck(node.Children[i] != -1, "All children of internal nodes are valid nodes.");

					const auto& curr_node = _nodes[node.Children[i]];
					if (!search_box.intersects(curr_node.Box))
						continue;

					find(node.Children[i], curr_node, search_box, validator, enclosed_points);
				}
			}
		}

	private:
		int highestBit(uint32_t n)
		{
			n |= (n >> 1);
			n |= (n >> 2);
			n |= (n >> 4);
			n |= (n >> 8);
			n |= (n >> 16);
			return n - (n >> 1);
		}

		int highestBit(uint64_t n)
		{
			n |= (n >> 1);
			n |= (n >> 2);
			n |= (n >> 4);
			n |= (n >> 8);
			n |= (n >> 16);
			n |= (n >> 32);
			return n - (n >> 1);
		}

		int msbIdx(uint32_t n)
		{
			auto v = highestBit(n);
			auto r = 0;

			while (v >>= 1)
			{
				r++;
			}

			return r;
		}

		int msbIdx(uint64_t n)
		{
			auto v = highestBit(n);
			auto r = 0;

			while (v >>= 1)
			{
				r++;
			}

			return r;
		}
	private:
		//! Maximum depth of the tree
		int _maxDepth;

		//! Size of cell at the finest level
		Eigen::Vector3f _cellSize;

		//! Top level node
		Node _topNode;

		//! List of all the nodes in the tree
		std::vector<Node> _nodes;

		//! Points managed in this tree
		std::vector<Eigen::Vector3f> _points;
	};
}}
