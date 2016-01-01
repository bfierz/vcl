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
#include <array>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Geometry
{
	// Cell types
	enum class CellType
	{
		None = -1,
		TetrahedralCell = 0,
		PyramidralCell = 1,
		PrismaCell = 2,
		HexahedralCell = 3
	};

	// Forward declaration
	template<typename T> struct CellTraits;

	template<typename Derived, typename IndexType>
	class Cell
	{
	public:
		static const int NumFaces     = CellTraits<Derived>::NumFaces;
		static const int NumVertices  = CellTraits<Derived>::NumVertices;
		static const int NumTriFaces  = CellTraits<Derived>::NumTriFaces;
		static const int NumQuadFaces = CellTraits<Derived>::NumQuadFaces;
		static const int NumEdges     = CellTraits<Derived>::NumEdges;

	public:
		using Traits = CellTraits<Derived>;
		using Vertices = std::array<IndexType, NumVertices> ;

		Cell() { static_assert(sizeof(Cell) == sizeof(Vertices) + sizeof(Element), "Constants don't occupy space."); }
		Cell(const Vertices& indices) : _indices(indices) {}
		
		IndexType& operator[](int index)       { return _indices[index]; }
		IndexType  operator[](int index) const { return _indices[index]; }
		
		IndexType& vertex(int index)       { return _indices[index]; }
		IndexType  vertex(int index) const { return _indices[index]; }

		Vertices& vertices() { return _indices; }
		const Vertices& vertices() const { return _indices; }
		
	private:
		Vertices _indices;
	};
	
	template<typename IndexType>
	class TetrahedralCell : public Cell<TetrahedralCell<IndexType>, IndexType>
	{
	public:
		using vector3_t   = Eigen::Matrix<float, 3, 1>;
		using vector4_t   = Eigen::Matrix<float, 4, 1>;
		using matrix3_t   = Eigen::Matrix<float, 3, 3>;
		using matrix4_t   = Eigen::Matrix<float, 4, 4>;
		using matrix3x4_t = Eigen::Matrix<float, 3, 4>;
		using matrix4x3_t = Eigen::Matrix<float, 4, 3>;

	public:
		TetrahedralCell() = default;
		TetrahedralCell(const std::array<IndexType, Cell<TetrahedralCell, IndexType>::NumVertices>& indices) : Cell<TetrahedralCell, IndexType>(indices) {}

	public:
		struct NaturalCoordinates3
		{
			float zeta_1, zeta_2, zeta_3;
			NaturalCoordinates3(float x, float y, float z) : zeta_1(x), zeta_2(y), zeta_3(z) {}
		};
		
		struct NaturalCoordinates4
		{
			float zeta_1, zeta_2, zeta_3, zeta_4;
			NaturalCoordinates4(float x, float y, float z, float w) : zeta_1(x), zeta_2(y), zeta_3(z), zeta_4(w) {}
		};
		
	public: // Shape functions
		// Compute the Jacobian matrix for the deformation field at a given integration
		// point.
		static matrix3_t jacobian(const matrix3x4_t& coords, const NaturalCoordinates3& gauss_point)
		{
			matrix3_t Jm = matrix3_t::Zero();
			for (int i = 0; i < 4; i++)
			{
				Jm.row(0) += dNi_zeta_1(i, gauss_point) * coords.col(i).transpose();
				Jm.row(1) += dNi_zeta_2(i, gauss_point) * coords.col(i).transpose();
				Jm.row(2) += dNi_zeta_3(i, gauss_point) * coords.col(i).transpose();
			}
			return Jm;
		}
		
		// Compute the spatial derivatives of the interpolation functions at a given
		// integration point
		static matrix3x4_t dNi_xyz(const matrix3_t& Jinv, const NaturalCoordinates3& gauss_point)
		{
			matrix3x4_t der = matrix3x4_t::Zero();
			vector3_t dNi_zeta_123;
			
			for (int i = 0; i < 4; i++)
			{
				dNi_zeta_123(0) = dNi_zeta_1(i, gauss_point);
				dNi_zeta_123(1) = dNi_zeta_2(i, gauss_point);
				dNi_zeta_123(2) = dNi_zeta_3(i, gauss_point);
				der.col(i) = Jinv * dNi_zeta_123;
			}

			return der;
		}
		
		// Compute the Jacobian matrix for the deformation field at a given integration
		// point.
		static matrix4_t jacobian(const matrix3x4_t& coords, const NaturalCoordinates4& gauss_point)
		{
			matrix4_t Jm = matrix4_t::Zero();
			for (int i = 0; i < 4; i++)
			{
				vector4_t vec(1.0, coords.col(i).x(), coords.col(i).y(), coords.col(i).z());
				Jm.col(0) += dNi_zeta_1(i, gauss_point) * vec;
				Jm.col(1) += dNi_zeta_2(i, gauss_point) * vec;
				Jm.col(2) += dNi_zeta_3(i, gauss_point) * vec;
				Jm.col(3) += dNi_zeta_4(i, gauss_point) * vec;
			}
			return Jm;
		}
		
		// Compute the spatial derivatives of the interpolation functions at a given
		// integration point
		static matrix4_t dNi_xyz(const matrix4_t& Jinv, const NaturalCoordinates4& gauss_point)
		{
			matrix4_t der = matrix4_t::Zero();
			vector4_t dNi_zeta_1234;
			
			for (int i = 0; i < 4; i++)
			{
				dNi_zeta_1234(0) = dNi_zeta_1(i, gauss_point);
				dNi_zeta_1234(1) = dNi_zeta_2(i, gauss_point);
				dNi_zeta_1234(2) = dNi_zeta_3(i, gauss_point);
				dNi_zeta_1234(3) = dNi_zeta_4(i, gauss_point);
				der.col(i) = Jinv * dNi_zeta_1234;
			}

			return der;
		}

	private:
		static float dNi_zeta_1(int i, const NaturalCoordinates3& gauss_point)
		{
			if (i == 0)
				return 1;
			else if (i == 3)
				return -1;
			else
				return 0;
		}
		
		static float dNi_zeta_2(int i, const NaturalCoordinates3& gauss_point)
		{
			if (i == 1)
				return 1;
			else if (i == 3)
				return -1;
			else
				return 0;
		}

		static float dNi_zeta_3(int i, const NaturalCoordinates3& gauss_point)
		{
			if (i == 2)
				return 1;
			else if (i == 3)
				return -1;
			else
				return 0;
		}
		
		static float dNi_zeta_1(int i, const NaturalCoordinates4& gauss_point)
		{
			if (i == 0)
				return 1;
			else
				return 0;
		}
		
		static float dNi_zeta_2(int i, const NaturalCoordinates4& gauss_point)
		{
			if (i == 1)
				return 1;
			else
				return 0;
		}

		static float dNi_zeta_3(int i, const NaturalCoordinates4& gauss_point)
		{
			if (i == 2)
				return 1;
			else
				return 0;
		}

		static float dNi_zeta_4(int i, const NaturalCoordinates4& gauss_point)
		{
			if (i == 3)
				return 1;
			else
				return 0;
		}
	};

	template<typename IndexType>
	struct CellTraits<TetrahedralCell<IndexType>>
	{
		static const int NumEdges = 6;
		static const int NumVertices = 4;
		static const int NumFaces = 4;
		static const int NumTriFaces = 4;
		static const int NumQuadFaces = 0;
	
		static const int edges[NumEdges][2];
		static const int quadFaces[1][4]; // NumQuadFaces is 0, which would result in a linker error, so setting to a dummy 1
		static const int triFaces[NumTriFaces][3];
	};
	
	template<typename IndexType>
	class PyramidralCell : public Cell<PyramidralCell<IndexType>, IndexType>
	{
	public:
		using vector3_t   = Eigen::Matrix<float, 3, 1>;
		using matrix3_t   = Eigen::Matrix<float, 3, 3>;
		using matrix3x5_t = Eigen::Matrix<float, 3, 5>;
		using matrix5x3_t = Eigen::Matrix<float, 5, 3>;

	public:
		PyramidralCell() = default;
		PyramidralCell(const std::array<IndexType, Cell<PyramidralCell, IndexType>::NumVertices>& indices) : Cell<PyramidralCell, IndexType>(indices) {}

	public:
		struct NaturalCoordinates
		{
			float xi, eta, zeta;
			NaturalCoordinates(float x, float y, float z) : xi(x), eta(y), zeta(z) {}
		};

	public: // Shape functions
		// Compute the Jacobian matrix for the deformation field at a given integration
		// point.
		static matrix3_t jacobian(const matrix3x5_t& coords, const NaturalCoordinates& gauss_point)
		{
			matrix3_t Jm = matrix3_t::Zero();
			for (int i = 0; i < 5; i++)
			{
				Jm.row(0) +=   dNi_xi(i, gauss_point) * coords.col(i).transpose();
				Jm.row(1) +=  dNi_eta(i, gauss_point) * coords.col(i).transpose();
				Jm.row(2) += dNi_zeta(i, gauss_point) * coords.col(i).transpose();
			}
			return Jm;
		}

		// Isoparametric coordinates of the nodes
		static NaturalCoordinates nodes(size_t i)
		{
			static NaturalCoordinates isoNodes[] = { NaturalCoordinates( 0, 0, 1),
													 NaturalCoordinates(-1,-1,-1),
													 NaturalCoordinates( 1,-1,-1),
													 NaturalCoordinates( 1, 1,-1),
													 NaturalCoordinates(-1, 1,-1) };
			return isoNodes[i];
		}

		static float shapeFunctionWeight(size_t i)
		{
			static float weights[] = { 1.0 / 2.0,
										 1.0 / 8.0,
									     1.0 / 8.0,
									     1.0 / 8.0,
									     1.0 / 8.0 };
			return weights[i];
		}

		// Isoparametric derivative wrt. xi of interpolation function N_i evaluated
		// at integration point gaussPoint
		static float dNi_xi(int i, const NaturalCoordinates& gauss_point)
		{
			NaturalCoordinates node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			float weight = shapeFunctionWeight(i);
			return float(weight*node.xi*(1+gauss_point.eta*node.eta)*(1+gauss_point.zeta*node.zeta));
		}
		
		// Isoparametric derivative wrt. eta of interpolation function N_i evaluated
		// at integration point gaussPoint
		static float dNi_eta(int i, const NaturalCoordinates& gauss_point)
		{
			NaturalCoordinates node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			float weight = shapeFunctionWeight(i);
			return float(weight*(1+gauss_point.xi*node.xi)*(node.eta)*(1+gauss_point.zeta*node.zeta));
		}

		// Isoparametric derivative wrt. zeta of interpolation function N_i evaluated
		// at integration point gaussPoint
		static float dNi_zeta(int i, const NaturalCoordinates& gauss_point)
		{
			NaturalCoordinates node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			float weight = shapeFunctionWeight(i);
			return float(weight*(1+gauss_point.xi*node.xi)*(1+gauss_point.eta*node.eta)*(node.zeta));
		}
		
		// Compute the spatial derivatives of the interpolation functions at a given
		// integration point
		static matrix3x5_t dNi_xyz(const matrix3_t& Jinv, const NaturalCoordinates& gauss_point)
		{
			matrix3x5_t der = matrix3x5_t::Zero();
			vector3_t dNi_xi_eta_zeta;
			
			for (int i = 0; i < 5; i++)
			{
				dNi_xi_eta_zeta(0) = dNi_xi(i, gauss_point);
				dNi_xi_eta_zeta(1) = dNi_eta(i, gauss_point);
				dNi_xi_eta_zeta(2) = dNi_zeta(i, gauss_point);
				der.col(i) = Jinv * dNi_xi_eta_zeta;
			}

			return der;
		}
	};
	
	template<typename IndexType>
	struct CellTraits<PyramidralCell<IndexType>>
	{
		static const int NumEdges = 8;
		static const int NumVertices = 5;
		static const int NumFaces = 5;
		static const int NumTriFaces = 4;
		static const int NumQuadFaces = 1;
	
		static const int edges[NumEdges][2];
		static const int quadFaces[NumQuadFaces][4];
		static const int triFaces[NumTriFaces][3];
	};

	template<typename IndexType>
	class PrismaCell : public Cell<PrismaCell<IndexType>, IndexType>
	{
	public:
		using vector3_t   = Eigen::Matrix<float, 3, 1>;
		using matrix3_t   = Eigen::Matrix<float, 3, 3>;
		using matrix3x6_t = Eigen::Matrix<float, 3, 6>;
		using matrix6x3_t = Eigen::Matrix<float, 6, 3>;

	public:
		struct NaturalCoordinates
		{
			float zeta_1, zeta_2, xi;
			NaturalCoordinates(float x, float y, float z) : zeta_1(x), zeta_2(y), xi(z) {}
		};

	public:
		PrismaCell() = default;
		PrismaCell(const std::array<IndexType, Cell<PrismaCell, IndexType>::NumVertices>& indices) : Cell<PrismaCell, IndexType>(indices) {}
		
	public: // Shape functions
		// Compute the Jacobian matrix for the deformation field at a given integration
		// point.
		static matrix3_t jacobian(const matrix3x6_t& coords, const NaturalCoordinates& gauss_point)
		{
			matrix3_t Jm = matrix3_t::Zero();
			for (int i = 0; i < 6; i++)
			{
				Jm.row(0) += dNi_zeta_1(i, gauss_point) * coords.col(i).transpose();
				Jm.row(1) += dNi_zeta_2(i, gauss_point) * coords.col(i).transpose();
				Jm.row(2) += dNi_xi(i, gauss_point) * coords.col(i).transpose();
			}
			return Jm;
		}
		
		// isoparametric coordinates of the nodes
		static int nodes(size_t i)
		{
			static int isoNodes[] = {-1,-1,-1,1,1,1};
			return isoNodes[i];
		}

		static float dNi_zeta_1(int i, const NaturalCoordinates& gauss_point)
		{
			int node = nodes(i);

			if (i == 0 || i == 3)
				return 1.0/2.0*(1.0 + node*gauss_point.xi);
			else if (i == 2 || i == 5)
				return -1.0/2.0*(1.0 + node*gauss_point.xi);
			else
				return 0;
		}
		
		static float dNi_zeta_2(int i, const NaturalCoordinates& gauss_point)
		{
			int node = nodes(i);

			if (i == 1 || i == 4)
				return 1.0/2.0*(1.0 + node*gauss_point.xi);
			else if (i == 2 || i == 5)
				return -1.0/2.0*(1.0 + node*gauss_point.xi);
			else
				return 0;
		}

		static float dNi_xi(int i, const NaturalCoordinates& gauss_point)
		{
			int node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			if (i == 0 || i == 3)
				return float(1.0/2.0*gauss_point.zeta_1*node);
			else if (i == 1 || i == 4)
				return float(1.0/2.0*gauss_point.zeta_2*node);
			else if (i == 2 || i == 5)
				return float(1.0/2.0*(1.0 - gauss_point.zeta_1 - gauss_point.zeta_2)*node);
			else
				return 0;
		}
		
		// Compute the spatial derivatives of the interpolation functions at a given
		// integration point
		static matrix3x6_t dNi_xyz(const matrix3_t& Jinv, const NaturalCoordinates& gauss_point)
		{
			matrix3x6_t der = matrix3x6_t::Zero();
			vector3_t dNi_zeta_12_xi;
			
			for (int i = 0; i < 6; i++)
			{
				dNi_zeta_12_xi(0) = dNi_zeta_1(i, gauss_point);
				dNi_zeta_12_xi(1) = dNi_zeta_2(i, gauss_point);
				dNi_zeta_12_xi(2) = dNi_xi(i, gauss_point);
				der.col(i) = Jinv * dNi_zeta_12_xi;
			}

			return der;
		}
	};

	template<typename IndexType>
	struct CellTraits<PrismaCell<IndexType>>
	{
		static const int NumEdges = 9;
		static const int NumVertices = 6;
		static const int NumFaces = 5;
		static const int NumTriFaces = 2;
		static const int NumQuadFaces = 3;
	
		static const int edges[NumEdges][2];
		static const int quadFaces[NumQuadFaces][4];
		static const int triFaces[NumTriFaces][3];
	};

	template<typename IndexType>
	class HexahedralCell : public Cell<HexahedralCell<IndexType>, IndexType>
	{
	public:
		using vector3_t   = Eigen::Matrix<float, 3, 1>;
		using matrix3_t   = Eigen::Matrix<float, 3, 3>;
		using matrix3x8_t = Eigen::Matrix<float, 3, 8>;
		using matrix8x3_t = Eigen::Matrix<float, 8, 3>;

	public:
		struct Node
		{
			float xi, eta, zeta;
			Node(float x, float y, float z) : xi(x), eta(y), zeta(z) {}
		};

	public:
		HexahedralCell() = default;
		HexahedralCell(const std::array<IndexType, Cell<HexahedralCell, IndexType>::NumVertices>& indices) : Cell<HexahedralCell, IndexType>(indices) {}

	public: // Shape functions
		static matrix3x8_t computeShapeFunctionGradients(const matrix3x8_t& coords, const vector3_t& gauss_point)
		{
			Node point(gauss_point[0], gauss_point[1], gauss_point[2]);
			matrix3_t J = jacobian(coords, point);
			matrix3_t Jinv = J.inverse();

			return dNi_xyz(Jinv, point);
		}

		static matrix8x3_t computeH(const Node& gauss_point)
		{
			matrix8x3_t H = matrix8x3_t::Zero();
			
			for (int i = 0; i < 8; i++)
			{
				H(i, 0) =   dNi_xi(i, gauss_point);
				H(i, 1) =  dNi_eta(i, gauss_point);
				H(i, 2) = dNi_zeta(i, gauss_point);
			}
			return H;
		}

		// Compute the Jacobian matrix for the deformation field at a given integration
		// point.
		static matrix3_t jacobian(const matrix3x8_t& coords, const Node& gauss_point)
		{
			matrix3_t Jm = matrix3_t::Zero();
			for (int i = 0; i < 8; i++)
			{
				Jm.row(0) +=   dNi_xi(i, gauss_point) * coords.col(i).transpose();
				Jm.row(1) +=  dNi_eta(i, gauss_point) * coords.col(i).transpose();
				Jm.row(2) += dNi_zeta(i, gauss_point) * coords.col(i).transpose();
			}
			return Jm;
		}

		// Isoparametric coordinates of the nodes
		static Node nodes(size_t i)
		{
			static Node isoNodes[] = { Node( 1, 1, 1),
									   Node( 1,-1, 1),
									   Node(-1,-1, 1),
									   Node(-1, 1, 1),
									   Node( 1, 1,-1),
									   Node( 1,-1,-1),
									   Node(-1,-1,-1),
									   Node(-1, 1,-1) };
			return isoNodes[i];
		}

		// Isoparametric derivative wrt. xi of interpolation function N_i evaluated
		// at integration point gaussPoint
		static float dNi_xi(int i, const Node& gauss_point)
		{
			Node node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			return float(1.0/8.0*node.xi*(1+gauss_point.eta*node.eta)*(1+gauss_point.zeta*node.zeta));
		}
		
		// Isoparametric derivative wrt. eta of interpolation function N_i evaluated
		// at integration point gaussPoint
		static float dNi_eta(int i, const Node& gauss_point)
		{
			Node node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			return float(1.0/8.0*(1+gauss_point.xi*node.xi)*(node.eta)*(1+gauss_point.zeta*node.zeta));
		}

		// Isoparametric derivative wrt. zeta of interpolation function N_i evaluated
		// at integration point gaussPoint
		static float dNi_zeta(int i, const Node& gauss_point)
		{
			Node node = nodes(i); // node = [xi_i, eta_i, zeta_i]
			return float(1.0/8.0*(1+gauss_point.xi*node.xi)*(1+gauss_point.eta*node.eta)*(node.zeta));
		}
		
		// Compute the spatial derivatives of the interpolation functions at a given
		// integration point
		static matrix3x8_t dNi_xyz(const matrix3_t& Jinv, const Node& gauss_point)
		{
			matrix3x8_t der = matrix3x8_t::Zero();
			vector3_t dNi_xi_eta_zeta;
			
			for (int i = 0; i < 8; i++)
			{
				dNi_xi_eta_zeta(0) = dNi_xi(i, gauss_point);
				dNi_xi_eta_zeta(1) = dNi_eta(i, gauss_point);
				dNi_xi_eta_zeta(2) = dNi_zeta(i, gauss_point);
				der.col(i) = Jinv * dNi_xi_eta_zeta;
			}

			return der;
		}
	};
	
	template<typename IndexType>
	struct CellTraits<HexahedralCell<IndexType>>
	{
		static const int NumEdges = 12;
		static const int NumVertices = 8;
		static const int NumFaces = 6;
		static const int NumTriFaces = 0;
		static const int NumQuadFaces = 6;

		static const int edges[NumEdges][2];
		static const int quadFaces[NumQuadFaces][4];
		static const int triFaces[1][3]; // NumTriFaces is 0, which would result in a linker error, so setting to a dummy 1
	};
	
	template<typename IndexType> const int CellTraits<TetrahedralCell<IndexType>>::edges[NumEdges][2] = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };
	template<typename IndexType> const int CellTraits<TetrahedralCell<IndexType>>::quadFaces[1][4] = { { -1, -1, -1, -1 } };
	template<typename IndexType> const int CellTraits<TetrahedralCell<IndexType>>::triFaces[NumTriFaces][3] = { { 1, 2, 3 }, { 2, 0, 3 }, { 3, 0, 1 }, { 0, 2, 1 } };
	
	template<typename IndexType> const int CellTraits<PyramidralCell<IndexType>>::edges[NumEdges][2] = { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 1 }, { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 } };
	template<typename IndexType> const int CellTraits<PyramidralCell<IndexType>>::quadFaces[NumQuadFaces][4] = { { 1, 2, 3, 4 } };
	template<typename IndexType> const int CellTraits<PyramidralCell<IndexType>>::triFaces[NumTriFaces][3] = { { 0, 1, 4 }, { 0, 2, 1 }, { 0, 3, 2 }, { 0, 4, 3 } };
	
	template<typename IndexType> const int CellTraits<PrismaCell<IndexType>>::edges[NumEdges][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 }, { 3, 4 }, { 4, 5 }, { 5, 3 }, { 0, 3 }, { 1, 4 }, { 2, 5 } };
	template<typename IndexType> const int CellTraits<PrismaCell<IndexType>>::quadFaces[NumQuadFaces][4] = { { 0, 3, 4, 1 }, { 0, 2, 5, 3 }, { 1, 4, 5, 2 } };
	template<typename IndexType> const int CellTraits<PrismaCell<IndexType>>::triFaces[NumTriFaces][3] = { { 0, 1, 2 }, { 5, 4, 3 } };

	template<typename IndexType> const int CellTraits<HexahedralCell<IndexType>>::edges[NumEdges][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 0, 3 }, { 4, 5 }, { 5, 6 }, { 6, 7 }, { 4, 7 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
	template<typename IndexType> const int CellTraits<HexahedralCell<IndexType>>::quadFaces[NumQuadFaces][4] = { { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 1, 2, 6, 5 }, { 0, 4, 7, 3 }, { 0, 3, 2, 1 }, { 4, 5, 6, 7 } };
	template<typename IndexType> const int CellTraits<HexahedralCell<IndexType>>::triFaces[1][3] = { {-1, -1, -1} };
}}
