/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <vcl/geometry/distanceTriangle3Triangle3.h>

// VCL
#include <vcl/core/simd/memory.h>

namespace Vcl { namespace Geometry {
#define clamp(v, a, b) max((a), min((v), (b)))

	template<typename T> auto inline dot(const T& p1, const T& p2) { return p1.dot(p2); }
	template<typename T> auto inline cross(const T& p1, const T& p2) { return p1.cross(p2); }
	

	template<typename Real, int Width>
	VectorScalar<bool, Width> project6(
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& ax,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& p1,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& p2,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& p3,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& q1,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& q2,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& q3)
	{
		using RealVec = VectorScalar<Real, Width>;

		RealVec P1 = dot(ax, p1);
		RealVec P2 = dot(ax, p2);
		RealVec P3 = dot(ax, p3);

		RealVec Q1 = dot(ax, q1);
		RealVec Q2 = dot(ax, q2);
		RealVec Q3 = dot(ax, q3);

		RealVec mx1 = max(P1, max(P2, P3));
		RealVec mn1 = min(P1, min(P2, P3));
		RealVec mx2 = max(Q1, max(Q2, Q3));
		RealVec mn2 = min(Q1, min(Q2, Q3));

		return (mn1 <= mx2) && (mn2 <= mx1);
	}

	template<typename Real, int Width>
	VectorScalar<bool, Width> closestEdgePoints(
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iTri1Pt,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iClosestPtToTri1,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iTri2Pt,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iClosestPtToTri2,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iSepDir)
	{
		using RealVec = VectorScalar<Real, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		RealVec3 awayDirection = iTri1Pt - iClosestPtToTri1;
		const RealVec isDiffDirection = dot(awayDirection, iSepDir);

		awayDirection = iTri2Pt - iClosestPtToTri2;
		const RealVec isSameDirection = dot(awayDirection, iSepDir);

		return (isDiffDirection <= RealVec{ 0 }) && (isSameDirection >= RealVec{ 0 });
	}
	
	//Code is taken from Real Time Collision Detection section 5.1.9 and has been adapted and changed to suit the paper's purposes.
	template<typename Real, int Width>
	VectorScalar<Real, Width> segmentSegmentSquared(
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oLine1Point,
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oLine2Point,
		const Segment<VectorScalar<Real, Width>, 3>& iLine1,
		const Segment<VectorScalar<Real, Width>, 3>& iLine2)
	{
		using RealVec = VectorScalar<Real, Width>;
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		const Real ulp = std::numeric_limits<Real>::epsilon();

		const RealVec3 dir1 = iLine1[1] - iLine1[0]; // Direction vector of segment S1
		const RealVec3 dir2 = iLine2[1] - iLine2[0]; // Direction vector of segment S2
		const RealVec3 r = iLine1[0] - iLine2[0];
		const RealVec a = dot(dir1, dir1); // Squared length of segment S1, always nonnegative
		RealVec e = dot(dir2, dir2); // Squared length of segment S2, always nonnegative
		const RealVec f = dot(dir2, r);
		const RealVec c = dot(dir1, r);
		const RealVec b = dot(dir1, dir2);

		// s and t are the parameter values form Line1 and iLine2 respectively. 
		RealVec s, t;
		//The following is always nonnegative.
		RealVec denom = a*e - b*b;
		// If segments not parallel, compute closest point on L1 to L2, and
		// clamp to segment S1. Else pick arbitrary s (here 0)

		//EVAN: As the previous description says if s can be arbitrary then we take the value given below instead of an if statement. 
		//To avoid a nonnegative denominator, we clip it. We know that it always has to be non-negative therefore we clip it with the following value.
		denom = max(denom, RealVec(ulp));
		s = clamp((b*f - c*e) / denom, RealVec(0), RealVec(1));
		// Compute point on L2 closest to S1(s) using
		// t = dot((P1+D1*s)-P2,D2) / dot(D2,D2) = (b*s + f) / e
		e = max(e, RealVec(ulp));
		t = (b*s + f) / e;
		// If t in [0,1] done. Else clamp t, recompute s for the new value
		// of t using s = dot((P2+D2*t)-P1,D1) / dot(D1,D1)= (t*b - c) / a
		// and clamp s to [0, 1]
		const RealVec newT = clamp(t, RealVec(0), RealVec(1));
		BoolVec mask = (newT != t);

		//Now test if all true or none true or some true. Use the select function to choose the respective values.
		s = select(mask, clamp((newT*b - c) / a, RealVec(0), RealVec(1)), s);

		oLine1Point = iLine1[0] + dir1 * s;
		oLine2Point = iLine2[0] + dir2 * newT;
		return (oLine1Point - oLine2Point).squaredNorm();
	}

	template<typename Real, int Width>
	VectorScalar<Real, Width> trianglePointSquared
	(
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTriPoint,
		const Triangle<VectorScalar<Real, Width>, 3>& iTri,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iPoint
	)
	{
		using RealVec = VectorScalar<Real, Width>;
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		// Check if P in vertex region outside A
		const RealVec3 ab = iTri[1] - iTri[0];
		const RealVec3 ac = iTri[2] - iTri[0];
		const RealVec3 ap = iPoint - iTri[0];
		const RealVec d1 = dot(ab, ap);
		const RealVec d2 = dot(ac, ap);
		const BoolVec mask1 = (d1 <= RealVec(0)) && (d2 <= RealVec(0));
		oTriPoint = iTri[0];
		BoolVec exit(mask1);
		if (all(exit))
			return (oTriPoint - iPoint).squaredNorm();  // barycentric coordinates (1,0,0)

		// Check if P in vertex region outside B
		const RealVec3 bp = iPoint - iTri[1];
		const RealVec d3 = dot(ab, bp);
		const RealVec d4 = dot(ac, bp);
		const BoolVec mask2 = (d3 >= RealVec(0)) && (d4 <= d3);
		// Closest point is the point iTri[1]. Update if necessary.
		oTriPoint = select<Real, Width, 3, 1>(exit, oTriPoint, select<Real, Width, 3, 1>(mask2, iTri[1], oTriPoint));
		exit |= mask2;
		if (all(exit))
			return (oTriPoint - iPoint).squaredNorm();  // barycentric coordinates (0,1,0)

		// Check if P in vertex region outside C
		const RealVec3 cp = iPoint - iTri[2];
		const RealVec d5 = dot(ab, cp);
		const RealVec d6 = dot(ac, cp);
		const BoolVec mask3 = (d6 >= RealVec(0)) && (d5 <= d6);
		// Closest point is the point iTri[2]. Update if necessary.
		oTriPoint = select<Real, Width, 3, 1>(exit, oTriPoint, select<Real, Width, 3, 1>(mask3, iTri[2], oTriPoint));
		exit |= mask3;
		if (all(exit))
			return (oTriPoint - iPoint).squaredNorm();  // barycentric coordinates (0,0,1)

		// Check if P in edge region of AB, if so return projection of P onto AB
		const RealVec vc = d1*d4 - d3*d2;
		const BoolVec mask4 = (vc <= RealVec(0)) && (d1 >= RealVec(0)) && (d3 <= RealVec(0));
		const RealVec v1 = d1 / (d1 - d3);
		const RealVec3 answer1 = iTri[0] + v1 * ab;		
		// Closest point is on the line ab. Update if necessary.
		oTriPoint = select<Real, Width, 3, 1>(exit, oTriPoint, select<Real, Width, 3, 1>(mask4, answer1, oTriPoint));
		exit |= mask4;
		if (all(exit))
			return (oTriPoint - iPoint).squaredNorm();  // barycentric coordinates (1-v,v,0)

		// Check if P in edge region of AC, if so return projection of P onto AC
		const RealVec vb = d5*d2 - d1*d6;
		const BoolVec mask5 = (vb <= RealVec(0)) && (d2 >= RealVec(0)) && (d6 <= RealVec(0));
		const RealVec w1 = d2 / (d2 - d6);
		const RealVec3 answer2 = iTri[0] + w1 * ac;
		// Closest point is on the line ac. Update if necessary.
		oTriPoint = select<Real, Width, 3, 1>(exit, oTriPoint, select<Real, Width, 3, 1>(mask5, answer2, oTriPoint));
		exit |= mask5;
		if (all(exit))
			return (oTriPoint - iPoint).squaredNorm();  // barycentric coordinates (1-w,0,w)

		// Check if P in edge region of BC, if so return projection of P onto BC
		const RealVec va = d3*d6 - d5*d4;
		const BoolVec mask6 = (va <= RealVec(0)) && ((d4 - d3) >= RealVec(0)) && ((d5 - d6) >= RealVec(0));
		RealVec w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		const RealVec3 answer3 = iTri[1] + w2 * (iTri[2] - iTri[1]);
		// Closest point is on the line bc. Update if necessary.
		oTriPoint = select<Real, Width, 3, 1>(exit, oTriPoint, select<Real, Width, 3, 1>(mask6, answer3, oTriPoint));
		exit |= mask6;
		if (all(exit))
			return (oTriPoint - iPoint).squaredNorm(); // barycentric coordinates (0,1-w,w)

		// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
		const RealVec denom = RealVec(1) / (va + vb + vc);
		const RealVec v2 = vb * denom;
		const RealVec w3 = vc * denom;
		const RealVec3 answer4 = iTri[0] + ab * v2 + ac * w3;
		const BoolVec mask7 = (answer4 - iPoint).squaredNorm() < (oTriPoint - iPoint).squaredNorm();
		// Closest point is inside triangle. Update if necessary.
		oTriPoint = select<Real, Width, 3, 1>(exit, oTriPoint, select<Real, Width, 3, 1>(mask7, answer4, oTriPoint));
		return (oTriPoint - iPoint).squaredNorm();  // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	}

	// Compute the distance between a triangle vertex and another triangle
	template<typename Real, int Width>
	VectorScalar<Real, Width> closestVertToTri(
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTriAPoint,
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTriBPoint,
		const Triangle<VectorScalar<Real, Width>, 3>& iTriA,
		const Triangle<VectorScalar<Real, Width>, 3>& iTriB)
	{
		using RealVec = VectorScalar<Real, Width>;
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		RealVec3 Ap, Bp, Cp;

		const RealVec A = trianglePointSquared(Ap, iTriA, iTriB[0]);
		const RealVec B = trianglePointSquared(Bp, iTriA, iTriB[1]);
		const RealVec C = trianglePointSquared(Cp, iTriA, iTriB[2]);

		const BoolVec AB = A < B;
		const RealVec ABdist = select(AB, A, B);
		const RealVec3 ABp = select<Real, Width, 3, 1>(AB, Ap, Bp);

		const BoolVec ABC = ABdist < C;
		oTriAPoint = select<Real, Width, 3, 1>(ABC, ABp, Cp);
		oTriBPoint = select<Real, Width, 3, 1>(ABC, select<Real, Width, 3, 1>(AB, iTriB[0], iTriB[1]), iTriB[2]);

		return select(ABC, ABdist, C);
	}

	template<typename Real, int Width>
	VectorScalar<Real, Width> closestEdgeToEdge(
		VectorScalar<bool, Width>& oIsFinished,
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTriAPoint,
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTriBPoint,
		const Segment<VectorScalar<Real, Width>, 3> iTriAEdges[3],
		const Segment<VectorScalar<Real, Width>, 3>& iTriBEdge,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& iTriBLastPt)
	{
		using RealVec = VectorScalar<Real, Width>;
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		//Test the triangle edge against all three edges of the triangle iTriA.
		RealVec3 A2p, A3p, B2p, B3p, separatingDir;

		const RealVec A = segmentSegmentSquared(oTriAPoint, oTriBPoint, iTriAEdges[0], iTriBEdge);
		//Test to see if the distances found so far were the closest:
		separatingDir = oTriBPoint - oTriAPoint;
		oIsFinished |= closestEdgePoints(iTriAEdges[1][0], oTriAPoint, iTriBLastPt, oTriBPoint, separatingDir);
		if (all(oIsFinished))
			return A;

		const RealVec B = segmentSegmentSquared(A2p, B2p, iTriAEdges[1], iTriBEdge);
		separatingDir = B2p - A2p;
		oIsFinished |= closestEdgePoints(iTriAEdges[2][0], A2p, iTriBLastPt, B2p, separatingDir);

		const BoolVec AB = A < B;
		const RealVec ABdist = select(AB, A, B);
		oTriAPoint = select<Real, Width, 3, 1>(AB, oTriAPoint, A2p);
		oTriBPoint = select<Real, Width, 3, 1>(AB, oTriBPoint, B2p);

		if (all(oIsFinished))
			return ABdist;

		const RealVec C = segmentSegmentSquared(A3p, B3p, iTriAEdges[2], iTriBEdge);
		separatingDir = B3p - A3p;
		oIsFinished |= closestEdgePoints(iTriAEdges[0][0], A3p, iTriBLastPt, B3p, separatingDir);

		const BoolVec ABC = ABdist < C;
		oTriAPoint = select<Real, Width, 3, 1>(ABC, oTriAPoint, A3p);
		oTriBPoint = select<Real, Width, 3, 1>(ABC, oTriBPoint, B3p);

		return select(ABC, ABdist, C);
	}

	template<typename Real, int Width>
	Eigen::Matrix<VectorScalar<Real, Width>, 3, 1> computeSeparatingDir(
		const Segment<VectorScalar<Real, Width>, 3>& iTri1Edges,
		const Segment<VectorScalar<Real, Width>, 3>& iTri2Edges)
	{
		using RealVec = VectorScalar<Real, Width>;
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		RealVec3 separatingDir = cross(iTri1Edges[1] - iTri1Edges[0], iTri2Edges[1] - iTri2Edges[0]);
		BoolVec directionMask = dot(separatingDir, iTri2Edges[0] - iTri1Edges[0]) < RealVec{ 0 };
		separatingDir = select(directionMask, separatingDir, -separatingDir);

		return separatingDir;
	}

	template<typename Real, int Width>
	VectorScalar<bool, Width> triContact(
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& P1,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& P2,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& P3,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& Q1,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& Q2,
		const Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& Q3)
	{
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;
		
		const RealVec3 p1 = { 0, 0, 0 }; //P1 - P1;
		const RealVec3 p2 = P2 - P1;
		const RealVec3 p3 = P3 - P1;

		const RealVec3 q1 = Q1 - P1;
		const RealVec3 q2 = Q2 - P1;
		const RealVec3 q3 = Q3 - P1;

		const RealVec3 e1 = P2 - P1;
		const RealVec3 e2 = P3 - P2;

		const RealVec3 f1 = Q2 - Q1;
		const RealVec3 f2 = Q3 - Q2;

		BoolVec mask(true);

		const RealVec3 n1 = cross(e1, e2);   mask &= project6(n1, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 m1 = cross(f1, f2);   mask &= project6(m1, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 ef11 = cross(e1, f1); mask &= project6(ef11, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 ef12 = cross(e1, f2); mask &= project6(ef12, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 f3 = q1 - q3;
		const RealVec3 ef13 = cross(e1, f3); mask &= project6(ef13, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 ef21 = cross(e2, f1); mask &= project6(ef21, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 ef22 = cross(e2, f2); mask &= project6(ef22, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 ef23 = cross(e2, f3); mask &= project6(ef23, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 e3 = p1 - p3;
		const RealVec3 ef31 = cross(e3, f1); mask &= project6(ef31, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 ef32 = cross(e3, f2); mask &= project6(ef32, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 ef33 = cross(e3, f3); mask &= project6(ef33, p1, p2, p3, q1, q2, q3); if (none(mask)) return false;
		const RealVec3 g1 = cross(e1, n1);   mask &= project6(g1, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 g2 = cross(e2, n1);   mask &= project6(g2, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 g3 = cross(e3, n1);   mask &= project6(g3, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 h1 = cross(f1, m1);   mask &= project6(h1, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 h2 = cross(f2, m1);   mask &= project6(h2, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;
		const RealVec3 h3 = cross(f3, m1);   mask &= project6(h3, p1, p2, p3, q1, q2, q3);   if (none(mask)) return false;

		return mask;
	}

	template<typename Real, int Width>
	VectorScalar<Real, Width> distanceImpl(
		const Triangle<VectorScalar<Real, Width>, 3>& iTri1,
		const Triangle<VectorScalar<Real, Width>, 3>& iTri2,
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTri1Point,
		Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>& oTri2Point)
	{
		using RealVec = VectorScalar<Real, Width>;
		using BoolVec = VectorScalar<bool, Width>;
		using RealVec3 = Eigen::Matrix<VectorScalar<Real, Width>, 3, 1>;

		//The three edges of the triangle. Keep orientation consistent.
		const Segment<RealVec, 3> tri1Edges[3] = { { iTri1[1], iTri1[0] },{ iTri1[2], iTri1[1] },{ iTri1[0], iTri1[2] } };
		const Segment<RealVec, 3> tri2Edges[3] = { { iTri2[1], iTri2[0] },{ iTri2[2], iTri2[1] },{ iTri2[0], iTri2[2] } };

		RealVec3 tri1Vector, tri2Vector;
		BoolVec  isFinished{ false };

		RealVec minDistsTriTri = closestEdgeToEdge(isFinished, oTri1Point, oTri2Point, tri1Edges, tri2Edges[0], iTri2[2]);
		if (all(isFinished))
			return minDistsTriTri;

		RealVec tmpMinDist = closestEdgeToEdge(isFinished, tri1Vector, tri2Vector, tri1Edges, tri2Edges[1], iTri2[0]);
		BoolVec mask = tmpMinDist < minDistsTriTri;
		minDistsTriTri = select(mask, tmpMinDist, minDistsTriTri);
		oTri1Point = select<Real, Width, 3, 1>(mask, tri1Vector, oTri1Point);
		oTri2Point = select<Real, Width, 3, 1>(mask, tri2Vector, oTri2Point);
		if (all(isFinished))
			return minDistsTriTri;

		tmpMinDist = closestEdgeToEdge(isFinished, tri1Vector, tri2Vector, tri1Edges, tri2Edges[2], iTri2[1]);
		mask = tmpMinDist < minDistsTriTri;
		minDistsTriTri = select(mask, tmpMinDist, minDistsTriTri);
		oTri1Point = select<Real, Width, 3, 1>(mask, tri1Vector, oTri1Point);
		oTri2Point = select<Real, Width, 3, 1>(mask, tri2Vector, oTri2Point);
		if (all(isFinished))
			return minDistsTriTri;

		// Now do vertex-triangle distances.
		tmpMinDist = closestVertToTri(tri2Vector, tri1Vector, iTri2, iTri1);
		mask = tmpMinDist < minDistsTriTri;
		oTri1Point = select<Real, Width, 3, 1>(mask, tri1Vector, oTri1Point);
		oTri2Point = select<Real, Width, 3, 1>(mask, tri2Vector, oTri2Point);
		minDistsTriTri = select(mask, tmpMinDist, minDistsTriTri);

		tmpMinDist = closestVertToTri(tri1Vector, tri2Vector, iTri1, iTri2);
		mask = tmpMinDist < minDistsTriTri;
		oTri1Point = select<Real, Width, 3, 1>(mask, tri1Vector, oTri1Point);
		oTri2Point = select<Real, Width, 3, 1>(mask, tri2Vector, oTri2Point);

		minDistsTriTri = select(mask, tmpMinDist, minDistsTriTri);
		//We need to rule out the triangles colliding with each other otherwise we can get a distance that is not equal to 0 although 
		//the true distance is 0. Hence we use simdTriContact here.

		BoolVec colliding{ triContact(iTri1[0], iTri1[1], iTri1[2], iTri2[0], iTri2[1], iTri2[2]) };
		return select(colliding, RealVec{ 0 }, minDistsTriTri);
	}
	
	float4 distance(const Triangle<float4, 3>& iTri1, const Triangle<float4, 3>& iTri2, Eigen::Matrix<float4, 3, 1>& oTri1Point, Eigen::Matrix<float4, 3, 1>& oTri2Point)
	{
		return distanceImpl<float, 4>(iTri1, iTri2, oTri1Point, oTri2Point);
	}

	float8 distance(const Triangle<float8, 3>& iTri1, const Triangle<float8, 3>& iTri2, Eigen::Matrix<float8, 3, 1>& oTri1Point, Eigen::Matrix<float8, 3, 1>& oTri2Point)
	{
		return distanceImpl<float, 8>(iTri1, iTri2, oTri1Point, oTri2Point);
	}

	float16 distance(const Triangle<float16, 3>& iTri1, const Triangle<float16, 3>& iTri2, Eigen::Matrix<float16, 3, 1>& oTri1Point, Eigen::Matrix<float16, 3, 1>& oTri2Point)
	{
		return distanceImpl<float, 16>(iTri1, iTri2, oTri1Point, oTri2Point);
	}
}}
