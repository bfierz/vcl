/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <algorithm>
#include <vector>

// GSL
#include <gsl/gsl>

// VCL
#include <vcl/core/contract.h>
#include <vcl/geometry/circle.h>
#include <vcl/geometry/halfedgemesh.h>
#include <vcl/geometry/segment.h>

namespace Vcl { namespace Geometry
{
	namespace Detail
	{
		struct DelaunayContext
		{
			//! Generated mesh conforming to a delaunay triangulation
			HalfEdgeMesh* Mesh;

			//! Points of the current region being processed
			gsl::span<const std::pair<Eigen::Vector2f, HalfEdgeMesh::VertexId>> Points;

			//! Edges constructed in the processed region
			std::vector<HalfEdgeMesh::EdgeId> Edges;

			//! Left-most vertex
			HalfEdgeMesh::HalfEdgeId LeftMost;

			//! Right-most vertex
			HalfEdgeMesh::HalfEdgeId RightMost;
		};

		std::array<HalfEdgeMesh::VertexId, 2> findConnectingEdgeCandidate(const DelaunayContext& left, const DelaunayContext& right)
		{
			VclRequire(left.Mesh == right.Mesh, "Mesh is the same");

			const auto& prev = [mesh = left.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Prev;
			};
			const auto& next = [mesh = left.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Next;
			};
			const auto& twin = [mesh = left.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Twin;
			};
			const auto& vertex = [mesh = left.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Vertex;
			};
			const auto& pos = [mesh = left.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->position(mesh->halfEdge(he_id).Vertex);
			};

			auto left_rightmost = left.RightMost;
			auto right_leftmost = right.LeftMost;
			const auto left_base = pos(left_rightmost);
			const auto right_base = pos(right_leftmost);

			HalfEdgeMesh::HalfEdgeId left_edge  = left_rightmost;
			HalfEdgeMesh::HalfEdgeId right_edge = right_leftmost;
			PointSegmentClass c_left = PointSegmentClass::Right;
			PointSegmentClass c_right = PointSegmentClass::Right;
			do
			{
				Eigen::Vector2f point_left  = pos(twin(prev(left_edge)));
				Eigen::Vector2f point_right = pos(twin(right_edge));

				c_left = classify({ left_base, right_base }, point_left);
				if (c_left == PointSegmentClass::Right)
				{
					left_edge = twin(prev(left_edge));
				}

				c_right = classify({ left_base, right_base }, point_right);
				if (c_right == PointSegmentClass::Right)
				{
					right_edge = next(twin(right_edge));
				}
			} while ((c_left == PointSegmentClass::Right && left_rightmost != left_edge) || (c_right == PointSegmentClass::Right && right_leftmost != right_edge));

			return{ vertex(left_edge), vertex(right_edge) };
		}

		HalfEdgeMesh::HalfEdgeId delaunayValidateLeft(DelaunayContext& ctx, HalfEdgeMesh::HalfEdgeId he_id)
		{
			const auto& halfedge = [mesh = ctx.Mesh](HalfEdgeMesh::EdgeId e_id)
			{
				return mesh->edge(e_id).HalfEdge;
			};
			const auto& prev = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Prev;
			};
			const auto& next = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Next;
			};
			const auto& twin = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Twin;
			};
			const auto& vertex = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Vertex;
			};
			const auto& pos = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->position(mesh->halfEdge(he_id).Vertex);
			};

			const auto g = pos(he_id);
			auto dg = he_id;

			const auto d = pos(twin(he_id));
			he_id = next(he_id);

			auto u = pos(twin(he_id));
			auto du = twin(he_id);

			auto v = pos(twin(next(he_id)));

			if (classify({ g, d }, u) == PointSegmentClass::Left)
			{
				/* 3 points aren't colinear */
				/* as long as the 4 points belong to the same circle, do the cleaning */
				while (v != d && v != g && isInCircle(g, d, u, v) == PointCircleClass::Inside)
				{
					const auto c = next(he_id);
					du = twin(next(he_id));
					ctx.Mesh->removeEdge(he_id);

					he_id = c;
					u = pos(du);
					v = pos(twin(next(he_id)));
				}

				if (v != d && v != g && isInCircle(g, d, u, v) == PointCircleClass::OnCircle)
				{
					du = prev(du);
					ctx.Mesh->removeEdge(he_id);
				}
			}
			else
			{
				// Points are co-linear
				du = dg;
			}

			return du;
		}

		HalfEdgeMesh::HalfEdgeId delaunayValidateRight(DelaunayContext& ctx, HalfEdgeMesh::HalfEdgeId he_id)
		{
			const auto& halfedge = [mesh = ctx.Mesh](HalfEdgeMesh::EdgeId e_id)
			{
				return mesh->edge(e_id).HalfEdge;
			};
			const auto& prev = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Prev;
			};
			const auto& next = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Next;
			};
			const auto& twin = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Twin;
			};
			const auto& vertex = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Vertex;
			};
			const auto& pos = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->position(mesh->halfEdge(he_id).Vertex);
			};

			he_id = twin(he_id);
			const auto rv = pos(he_id);
			const auto dd = he_id;
			const auto lv = pos(twin(he_id));
			he_id = prev(he_id);
			auto u = pos(twin(he_id));
			auto du = twin(he_id);

			auto v = pos(twin(prev(he_id)));

			if (classify({ lv, rv }, u) == PointSegmentClass::Left)
			{
				while (v != lv && v != rv && isInCircle(lv, rv, u, v) == PointCircleClass::Inside)
				{
					auto c = prev(he_id);
					du = twin(c);
					ctx.Mesh->removeEdge(he_id);
					he_id = c;
					u = pos(du);
					v = pos(twin(prev(he_id)));
				}

				if (v != lv && v != rv && isInCircle(lv, rv, u, v) == PointCircleClass::OnCircle)
				{
					du = next(du);
					ctx.Mesh->removeEdge(he_id);
				}
			}
			else
			{
				du = dd;
			}

			return du;
		}

		HalfEdgeMesh::HalfEdgeId delaunayValidate(DelaunayContext& ctx, HalfEdgeMesh::HalfEdgeId he_id)
		{
			const auto& halfedge = [mesh = ctx.Mesh](HalfEdgeMesh::EdgeId e_id)
			{
				return mesh->edge(e_id).HalfEdge;
			};
			const auto& twin = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Twin;
			};
			const auto& vertex = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Vertex;
			};
			const auto& pos = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->position(mesh->halfEdge(he_id).Vertex);
			};

			const auto g = pos(he_id);
			auto gd = delaunayValidateLeft(ctx, he_id);
			auto g_p = pos(gd);

			const auto d = pos(twin(he_id));
			auto dd = delaunayValidateRight(ctx, he_id);
			auto d_p = pos(dd);

			if (g != g_p && d != d_p)
			{
				PointCircleClass a = isInCircle(g, d, g_p, d_p);
				if (a != PointCircleClass::OnCircle)
				{
					if (a == PointCircleClass::Inside)
					{
						g_p = g;
						gd = he_id;
					}
					else
					{
						d_p = d;
						dd = twin(he_id);
					}
				}
			}

			return halfedge(ctx.Mesh->addEdge({ vertex(gd), vertex(dd) }));
		}

		void delaunayMerge(DelaunayContext& ctx, DelaunayContext& left, DelaunayContext& right)
		{
			const auto& halfedge = [mesh = ctx.Mesh](HalfEdgeMesh::EdgeId e_id)
			{
				return mesh->edge(e_id).HalfEdge;
			};
			const auto& prev = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Prev;
			};
			const auto& next = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Next;
			};
			const auto& twin = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Twin;
			};
			const auto& vertex = [mesh = ctx.Mesh](HalfEdgeMesh::HalfEdgeId he_id)
			{
				return mesh->halfEdge(he_id).Vertex;
			};
			const auto& pos = [mesh = ctx.Mesh](HalfEdgeMesh::VertexId v_id)
			{
				return mesh->position(v_id);
			};
			
			const auto left_most_vert = vertex(left.LeftMost);
			const auto right_most_vert = vertex(right.RightMost);

			// Find the initial edge candidate
			const auto cand_vert = findConnectingEdgeCandidate(left, right);

			auto cand_edge = halfedge(ctx.Mesh->addEdge(cand_vert));
			auto u = vertex(twin(next(cand_edge)));
			auto v = vertex(twin(prev(twin(cand_edge))));

			while (classify({ pos(cand_vert[0]), pos(cand_vert[1]) }, pos(u)) == PointSegmentClass::Left ||
				   classify({ pos(cand_vert[0]), pos(cand_vert[1]) }, pos(v)) == PointSegmentClass::Left)
			{
				cand_edge = delaunayValidate(ctx, cand_edge);
				u = vertex(twin(next(cand_edge)));
				v = vertex(twin(prev(twin(cand_edge))));
			}

			right.RightMost = ctx.Mesh->vertex(right_most_vert).HalfEdge;
			left.LeftMost   = ctx.Mesh->vertex(left_most_vert).HalfEdge;

			while (classify({ pos(vertex(right.RightMost)), pos(vertex(twin(right.RightMost))) }, pos(vertex(twin(prev(right.RightMost))))) == PointSegmentClass::Right)
				right.RightMost = prev(right.RightMost);

			while (classify({ pos(vertex(left.LeftMost)), pos(vertex(twin(left.LeftMost))) }, pos(vertex(twin(prev(left.LeftMost))))) == PointSegmentClass::Right)
				left.LeftMost = prev(left.LeftMost);

			ctx.LeftMost = left.LeftMost;
			ctx.RightMost = right.RightMost;
		}

		void delaunaySplit(DelaunayContext& ctx)
		{
			VclRequire(ctx.Points.size() > 1, "At least 2 points are supplied.");

			auto points = ctx.Points;
			if (points.size() > 3)
				// Divide and conquer
			{
				const size_t split = points.size() / 2;

				DelaunayContext left;
				left.Mesh = ctx.Mesh;
				left.Points = points.subspan(0, split);
				DelaunayContext right;
				right.Mesh = ctx.Mesh;
				right.Points = points.subspan(split, points.size() - split);

				delaunaySplit(left);
				delaunaySplit(right);
				delaunayMerge(ctx, left, right);
			}
			else if (points.size() == 3)
				// Make a single triangle
			{
				// Check triangle winding and force ccw
				if (classify({points[0].first, points[1].first}, points[2].first) == PointSegmentClass::Left)
				{
					ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[0].second, points[1].second }));
					ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[1].second, points[2].second }));
					ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[2].second, points[0].second }));
					ctx.LeftMost  = ctx.Mesh->vertex(points[2].second).HalfEdge;
					ctx.RightMost = ctx.Mesh->vertex(points[0].second).HalfEdge;
				}
				else
				{
					ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[0].second, points[2].second }));
					ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[2].second, points[1].second }));
					ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[1].second, points[0].second }));
					ctx.LeftMost  = ctx.Mesh->vertex(points[2].second).HalfEdge;
					ctx.RightMost = ctx.Mesh->vertex(points[0].second).HalfEdge;
				}
			}
			else if (points.size() == 2)
				// Make a single edge
			{
				ctx.Edges.emplace_back(ctx.Mesh->addEdge({ points[0].second, points[1].second }));
				ctx.LeftMost  = ctx.Mesh->vertex(points[0].second).HalfEdge;
				ctx.RightMost = ctx.Mesh->vertex(points[1].second).HalfEdge;
			}
		}
	}

	// Modeled accordign to
	// http://www.geom.uiuc.edu/~samuelp/del_project.html
	HalfEdgeMesh computeDelaunayTriangulation(gsl::span<const Eigen::Vector2f> points)
	{
		std::vector<std::pair<Eigen::Vector2f, HalfEdgeMesh::VertexId>> delaunay_points;
		delaunay_points.reserve(points.size());

		// Store the index along side with the points
		unsigned int num_points = 0;
		std::transform(std::begin(points), std::end(points), std::back_inserter(delaunay_points), [&num_points](const Eigen::Vector2f& pt)
		{
			return std::make_pair(pt, HalfEdgeMesh::VertexId{ num_points++ });
		});
		// Sort along the x-axis
		std::sort(std::begin(delaunay_points), std::end(delaunay_points), [](const auto& pt_a, const auto& pt_b)
		{
			if (pt_a.first.x() < pt_b.first.x())
				return true;
			else if (pt_a.first.x() > pt_b.first.x())
				return false;
			if (pt_a.first.y() < pt_b.first.y())
				return true;
			else if (pt_a.first.y() > pt_b.first.y())
				return false;

			VclCheck(false, "Two points should not be equal.");
			return false;
		});

		if (num_points >= 3)
		{
			HalfEdgeMesh delaunay_mesh;
			delaunay_mesh.addVertices(points);

			Detail::DelaunayContext ctx;
			ctx.Mesh = &delaunay_mesh;
			ctx.Points = delaunay_points;
			Detail::delaunaySplit(ctx);

			return delaunay_mesh;
		}
	}
}}
