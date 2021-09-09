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
#include <absl/types/span.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/util/mortoncodes.h>

namespace Vcl { namespace Core
{
	// http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
	// http://devblogs.nvidia.com/parallelforall/wp-content/uploads/sites/3/2012/11/karras2012hpg_paper.pdf
	// EG 2009 - Lauterbach, Garland, Sengupta, Luebke, Manocha - Fast BVH Construction on GPUs (http://luebke.us/publications/eg09.pdf)
	// EG UK Computer Graphics & Visual Computing (2014) - Apetrei - Fast and Simple Agglomerative LBVH Construction
	// https://dcgi.fel.cvut.cz/projects/emc/emc2017.pdf
	class BVH
	{
	private:
		struct Node
		{
			//! Axis-aligned bounding box
			Eigen::AlignedBox3f Box;

			//! Is leaf-node
			uint64_t Leaf : 1;

			// Morton-code of the node
			uint64_t Prefix : 63;

			// Link to the parent
			int Parent;

			union
			{
				int Left;
				int First;
			};

			union
			{
				int Right;
				int Last;
			};
		};

	public:
		BVH(float cell_size)
		: _cellSize(cell_size)
		, _boundingBox(Eigen::Vector3f{ 0, 0, 0 }, Eigen::Vector3f{ 1024 * _cellSize, 1024 * _cellSize, 1024 * _cellSize })
		{
			VclRequire(cell_size > 0, "Size is valid.");
		}

	public:
		void clear()
		{
			// Delete old tree
			_nodes.clear();

			// Delete the allocated morton codes
			_sortedMortonCodes.clear();
		}

		void assign(const std::vector<Eigen::Vector3f>& points, bool build_recursive = true)
		{
			using Vcl::Mathematics::clz64;
			using Vcl::Util::splitHighestBit;

			// Clear the current tree
			clear();

			// Early out
			if (points.size() == 0)
				return;

			_points = points;
			_sortedMortonCodes =
				prepareMortonCodes(points, _boundingBox, _cellSize);

			const auto same_codes = collectSameCodes(_sortedMortonCodes);
			const auto& ranges = same_codes.first;
			const auto& collected_codes = same_codes.second;

			if (build_recursive)
			{
				// Preallocate nodes
				_nodes.reserve(2 * ranges.size());

				auto root = generateHierarchy(_sortedMortonCodes, 0, _sortedMortonCodes.size() - 1);
			}
			else
			{
				// Preallocate nodes
				_nodes.resize(2 * ranges.size());

				// Base index for leafs.
				// First half contains the internal nodes.
				size_t base_leaf_idx = ranges.size();

				// Store the objects in the leaf-nodes.
				for (size_t idx = 0; idx < ranges.size(); idx++)
				{
					Node& n = _nodes[base_leaf_idx + idx];
					n.Leaf = 1;
					n.First = ranges[idx].first;
					n.Last = ranges[idx].second;
				}

				// Construct the inner tree nodes.
				// Stored in the first half of the array
				for (size_t idx = 0; idx < ranges.size() - 1; idx++)
				{
					// 'range' contains the start, end of the objects belonging
					// to the current node
					auto range = determineRange(collected_codes, idx);
					int first = range.first;
					int last = range.second;

					// Given 'range', determine where to split the node
					const int split = splitHighestBit<int>(collected_codes, first, last);

					int left;
					if (split == first)
					{
						// No 'left' nodes, must be a leaf
						left = base_leaf_idx + split;
					}
					else
					{
						left = split;
					}

					int right;
					if (split + 1 == last)
					{
						// No 'right' nodes, must be a leaf
						right = base_leaf_idx + split + 1;
					}
					else
					{
						right = split + 1;
					}

					Node& n = _nodes[idx];
					n.Leaf = 0;

					const int split_prefix = clz64(collected_codes[split].first ^ collected_codes[split + 1].first);
					n.Prefix = 1 << (63 - split_prefix);

					// Record parent-child relationships
					n.Left = left;
					n.Right = right;
					_nodes[left].Parent = idx;
					_nodes[right].Parent = idx;
				}

				// Verify tree
				VclAssertBlock
				{
					// Running counter
					uint64_t curr_code{ 0 };

					// Node stack
					std::vector<int> stack;

					stack.push_back(0);
					while (!stack.empty())
					{
						int top = stack.back();
						stack.pop_back();

						auto curr_node = &_nodes[top];

						// Check if we found a leaf
						if (top >= base_leaf_idx)
						{
							uint64_t new_code = _sortedMortonCodes[_nodes[top].First].first;
							VclCheck(implies(curr_code > 0, new_code > curr_code), "Codes are read with increasing magnitude.");

							curr_code = new_code;
						}
						else
						{
							stack.push_back(_nodes[top].Right);
							stack.push_back(_nodes[top].Left);
						}
					}
				}
			}
		}

		size_t find(const Eigen::AlignedBox3f& search_box, std::vector<int>& enclosed_points)
		{
			const auto validator = [search_box](const Eigen::Vector3f& p) { return search_box.contains(p); };
			find(0, search_box, validator, enclosed_points);
			return enclosed_points.size();
		}

		size_t find(Vector3f p, float radius, std::vector<int>& enclosed_points)
		{
			const auto validator = [center = p, r2 = radius * radius](const Eigen::Vector3f& p)
			{ return (p - center).squaredNorm() <= r2; };

			Eigen::AlignedBox3f search_box{ p - Eigen::Vector3f::Constant(radius), p + Eigen::Vector3f::Constant(radius) };
			find(0, search_box, validator, enclosed_points);
			return enclosed_points.size();

			// Compute the size of the finest octree cell
			float inv_cell_size{ 1 / _cellSize };

			// Compute morton code of point
			Eigen::Vector3f scaled_p = (p - _boundingBox.min()) * inv_cell_size;
			uint64_t code = Util::MortonCode::encode(scaled_p.x(), scaled_p.y(), scaled_p.z());

			// Compute range of radius
			uint64_t nr_cells = (uint64_t) ceil(radius * inv_cell_size);

			// Number of blocks with 8 cells
			nr_cells /= 2;

			// Escape bit-pattern to stop tree traversal
			uint64_t collect_pattern = nr_cells * 8;

			// Exit condition
			size_t base = _nodes.size() / 2;
			
			size_t curr_node = 0;
			auto split = _nodes[curr_node].Prefix;
			while (curr_node < base && split >= collect_pattern)
			{
				if ((code & split) != 0)
				{
					curr_node = _nodes[curr_node].Right;
				}
				else
				{
					curr_node = _nodes[curr_node].Left;
				}

				split = _nodes[curr_node].Prefix;
			}

			return 0;
		}

	private:
		size_t generateHierarchy(const std::vector<std::pair<uint64_t, int>>& codes, int first, int last)
		{
			using Vcl::Util::splitHighestBit;

			// Single object => create a leaf node.
			const auto first_code = codes[first].first;
			const auto last_code = codes[last].first;
			if (first_code == last_code)
			{
				Node n;
				n.Box = Vcl::Util::MortonCode::calculateBoundingBox(first_code, _cellSize);
				n.Leaf = 1;
				n.First = first;
				n.Last = last;

				_nodes.emplace_back(n);
				return _nodes.size() - 1;
			}

			// Inner node
			const auto node_idx = _nodes.size();
			_nodes.emplace_back(Node{});
			Node& n = _nodes.back();
			n.Leaf = 0;
			
			// Determine where to split the range
			int split = splitHighestBit<int>(codes, first, last);

			// Process the resulting sub-ranges recursively
			auto left = generateHierarchy(codes, first, split);
			auto right = generateHierarchy(codes, split + 1, last);
			n.Box = _nodes[left].Box;
			n.Box.extend(_nodes[right].Box);
			n.Left = left;
			n.Right = right;

			return node_idx;
		}

		static std::vector<std::pair<uint64_t, int>>
			prepareMortonCodes(absl::Span<const Eigen::Vector3f> points, const Eigen::AlignedBox3f& bb, float cell_size)
		{
			// Compute the size of the finest octree cell
			const float inv_cell_size{ 1 / cell_size };

			// Compute the morton code for each point and link it to the original position
			std::vector<std::pair<uint64_t, int>> codes;
			codes.resize(points.size());
			for (int i = 0; i < codes.size(); i++)
			{
				const Eigen::Vector3f p = (points[i] - bb.min()) * inv_cell_size;
				codes[i] = std::make_pair(Util::MortonCode::encode(p.x(), p.y(), p.z()), i);
			}

			// Sort the codes in order to obtain a space filling curve
			std::sort(codes.begin(), codes.end(), [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b)
			{
				return a.first < b.first;
			});

			return codes;
		}

		//! Create a list of leafs, which contain begin/end pointers for each cell.
		//! Additionally, create a list with the number of elements per cell.
		static 
			std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<uint64_t, int>>>
			collectSameCodes(absl::Span<const std::pair<uint64_t, int>> sorted_codes)
		{
			std::vector<std::pair<int, int>> ranges;
			ranges.reserve(sorted_codes.size());

			std::vector<std::pair<uint64_t, int>> counts;
			counts.reserve(sorted_codes.size());

			// Start with first value
			auto curr_code = sorted_codes.front().first;
			int curr_range_begin = 0;
			int curr_range_end = 0;

			for (const auto& code_n_idx : sorted_codes)
			{
				if (code_n_idx.first == curr_code)
				{
					curr_range_end++;
				}
				else
				{
					// Found a new morton code, store the old range
					ranges.emplace_back(curr_range_begin, curr_range_end);
					counts.emplace_back(curr_code, curr_range_end - curr_range_begin);

					// Begin a new range
					curr_range_begin = curr_range_end;
					curr_range_end++;

					curr_code = code_n_idx.first;
				}
			}

			// Add the last range
			ranges.emplace_back(curr_range_begin, curr_range_end);
			counts.emplace_back(curr_code, curr_range_end - curr_range_begin);

			return{std::move(ranges), std::move(counts)};
		}

		//! Find the range of objects covered by node \p index
		//! \returns The begin and end of the covered objects
		//! Implemented according to https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
		std::pair<int, int> determineRange(absl::Span<const std::pair<uint64_t, int>> codes, int index)
		{
			VclRequire(std::is_sorted(std::begin(codes), std::end(codes),
				[](const auto& a, const auto& b) { return a.first < b.first; }),
				"Morton codes are sorted");
			VclRequire(std::adjacent_find(std::begin(codes), std::end(codes),
				[](const auto& a, const auto& b) { return a.first == b.first; }),
				"Morton codes are unique");

			using Vcl::Mathematics::clz64;
			using Vcl::Mathematics::sgn;

			// Determine search direction (+1, -1)
			const auto minus_one = codes[index - 1].first;
			const auto current = codes[index + 0].first;
			const auto plus_one = codes[index + 1].first;

			const auto prefix_minus = clz64(minus_one ^ current);
			const auto prefix_plus  = clz64(current ^ plus_one);

			const std::pair<int, uint64_t> dir_min = [](uint64_t a, uint64_t b)
			{
				if (a > b)
					return std::make_pair(-1, b);
				else
					return std::make_pair(+1, a);
			}(prefix_minus, prefix_plus);
			const int dir = dir_min.first;

			// Search the upper bound and lengh of search
			const auto d_min = dir_min.second;
			uint64_t l_max = 2;
			uint64_t search_idx = index + l_max * dir;
			while ((0 <= search_idx && search_idx < codes.size()) ? 
				(clz64(current ^ codes[search_idx].first) > d_min) : false)
			{
				l_max *= 2;
				search_idx = index + l_max * dir;
			}

			// Find the other end using binary search.
			// l_max/2, l_max/4, l_max/8, ...
			int64_t l = 0;
			for (int64_t t = l_max; t >= 1; t /= 2)
			{
				const auto new_index = index + (l + t)*dir;
				if (0 <= new_index && new_index < codes.size() && clz64(current ^ codes[new_index].first) > d_min)
					l += t;
			}

			// Return the range depending on the direction.
			// Finding the correct split position is performed by the caller.
			if (dir == 1)
				return std::make_pair(index, index + l * dir);
			else
				return std::make_pair(index + l * dir, index);

			/*using Vcl::Mathematics::clz64;

			//so we don't have to call it every time
			int lso = codes.size() - 1;
			//tadaah, it's the root node
			if (index == 0)
				return std::make_pair(0, lso);
			//direction to walk to, 1 to the right, -1 to the left
			int dir;
			//morton code diff on the outer known side of our range ... diff mc3 diff mc4 ->DIFF<- [mc5 diff mc6 diff ... ] diff .. 
			int d_min;
			int initialindex = index;

			auto minone = codes[index - 1].first;
			auto precis = codes[index + 0].first;
			auto pluone = codes[index + 1].first;
			if ((minone == precis && pluone == precis))
			{
				//set the mode to go towards the right, when the left and the right
				//object are being the same as this one, so groups of equal
				//code will be processed from the left to the right
				//and in node order from the top to the bottom, with each node X (ret.x = index)
				//containing Leaf object X and nodes from X+1 (the split func will make this split there)
				//till the end of the groups
				//(if any bit differs... DEP=32) it will stop the search
				while (index > 0 && index < lso)
				{
					//move one step into our direction
					index += 1;
					if (index >= lso)
						//we hit the left end of our list
						break;

					if (codes[index].first != codes[index + 1].first)
						//there is a diffrence
						break;
				}
				//return the end of equal grouped codes
				return std::make_pair(initialindex, index);
			}
			else
			{
				//Our codes differ, so we seek for the ranges end in the binary search fashion:
				auto lr = std::make_pair(clz64(precis ^ minone), clz64(precis ^ pluone));
				//now check wich one is higher (codes put side by side and wrote from up to down)
				if (lr.first > lr.second)
				{//to the left, set the search-depth to the right depth
					dir = -1;
					d_min = lr.second;
				}
				else {//to the right, set the search-depth to the left depth
					dir = 1;
					d_min = lr.first;
				}
			}

			//Now look for an range to search in (power of two)
			int l_max = 2;
			//so we don't have to calc it 3x
			int testindex = index + l_max * dir;
			while ((testindex <= lso && testindex >= 0) ? (clz64(precis ^ codes[testindex].first) > d_min) : (false))
			{
				l_max *= 2; testindex = index + l_max * dir;
			}
			int l = 0;
			//go from l_max/2 ... l_max/4 ... l_max/8 .......... 1 all the way down
			for (int div = 2; l_max / div >= 1; div *= 2)
			{
				//calculate the ofset state
				int t = l_max / div;
				//calculate where to test next
				int newTest = index + (l + t)*dir;
				//test if in code range
				if (newTest <= lso && newTest >= 0)
				{
					int splitPrefix = clz64(precis ^ codes[newTest].first);
					//and if the code is higher then our minimum, update the position
					if (splitPrefix > d_min)
						l = l + t;
				}
			}

			//now give back the range (in the right order, [lower|higher])
			if (dir == 1)
				return std::make_pair(index, index + l * dir);
			else
				return std::make_pair(index + l * dir, index);*/
		}

		template<typename Func>
		void find
		(
			unsigned int node_idx,
			const Eigen::AlignedBox3f& box,
			const Func& validator,
			std::vector<int>& enclosed_points
		)
		{
			const auto& node = _nodes[node_idx];

			if (node.Box.intersects(box))
			{
				if (node.Leaf != 0)
				{
					auto beg = _sortedMortonCodes.begin() + node.First;
					const auto end = _sortedMortonCodes.begin() + node.Last + 1;
					for (; beg != end; ++beg)
					{
						if (validator(_points[beg->second]))
							enclosed_points.emplace_back(beg->second);
					}
				}
				else
				{
					find(node.Left, box, validator, enclosed_points);
					find(node.Right, box, validator, enclosed_points);
				}
			}
		}

	public:
		const std::vector<Node>& nodes() const { return _nodes; }
		const std::vector<std::pair<uint64_t, int>>& sortedMortonCodes() const { return _sortedMortonCodes; }
		
	private:
		//! Size of a cell
		float _cellSize{ 0 };

		//! Bounding box
		Eigen::AlignedBox3f _boundingBox;

		//! List of all the nodes in the tree.
		//! First N-1 entries are the internal nodes,
		//! the next N nodes are the leafs.
		std::vector<Node> _nodes;

		//! Sorted morton codes and links to the actual data
		std::vector<std::pair<uint64_t, int>> _sortedMortonCodes;

		//! Stored points
		std::vector<Eigen::Vector3f> _points;
	};
}}
