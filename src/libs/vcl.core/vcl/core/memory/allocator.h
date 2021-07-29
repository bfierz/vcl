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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#if (defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64)) && !defined(VCL_ABI_WINAPI)
#include <mm_malloc.h> // Required for _mm_malloc
#endif
#include <stddef.h>    // Required for size_t and ptrdiff_t and NULL
#include <stdlib.h>    // Required for aligned_alloc
#include <limits>      // Required for numeric_limits
#include <memory>      // Required for std::allocator
#include <new>         // Required for placement new and std::bad_alloc
#include <stdexcept>   // Required for std::length_error

// VCL
#include <vcl/core/contract.h>

/*!
 * Picking up an idea formulated in:
 * http://www.codeproject.com/Articles/4795/C-Standard-Allocator-An-Introduction-and-Implement
 * http://jrruethe.github.io/blog/2015/11/22/allocators/
 * http://upcoder.com/5/zero-initialisation-for-classes
 */
namespace Vcl { namespace Core
{
	template<typename T>
	class ObjectTraits
	{
	public: // Typedefs
		using value_type = T;
		using pointer = value_type*;

	public:
		//! Convert an ObjectTraits<T> to ObjectTraits<U>
		template<typename U>
		struct rebind
		{
			using other = ObjectTraits<U>;
		};

	public:
		//! Default constructor
		explicit ObjectTraits() noexcept = default;

		//! Conversion constructor
		template <typename U>
		explicit ObjectTraits(ObjectTraits<U> const&) {}

		//! Compute address of an object
		T* address(T& r) { return std::addressof(r); }

		//! Compute address of an object
		T const* address(T const& r) { return std::addressof(r); }

		//! Call constructor of p of type U
		template<typename U, typename... Args>
		void construct(U* p, Args&&... args)
		{
			::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
		}

		//! Call the destructor of p of type U
		template<typename U>
		void destroy(U* p)
		{
			// Fixes a GCC warning
			VCL_UNREFERENCED_PARAMETER(p);
			p->~U();
		}
	};
	
	//! Object trait avoiding initialization of objects
	template<typename T>
	class NoInitObjectTraits
	{
	public: // Typedefs
		using value_type = T;
		using pointer = value_type*;

	public:
		//! Convert an NoInitObjectTraits<T> to NoInitObjectTraits<U>
		template<typename U>
		struct rebind
		{
			using other = NoInitObjectTraits<U>;
		};

	public:
		//! Default constructor
		explicit NoInitObjectTraits() noexcept = default;

		//! Conversion constructor
		template <typename U>
		explicit NoInitObjectTraits(NoInitObjectTraits<U> const&) {}

		//! Compute address of an object
		T* address(T& r) { return std::addressof(r); }

		//! Compute address of an object
		T const* address(T const& r) { return std::addressof(r); }

		//! Call constructor of p of type U
		template<typename U, typename... Args>
		void construct(U* p, Args&&...)
		{
			// Omit the '()' to avoid initialization of the objects
			::new(static_cast<void*>(p)) U;
		}

		//! Provide the object destruction interface
		//! As no initialiation happed, no destruction is performed
		template<typename U>
		void destroy(U* p)
		{
			VCL_UNREFERENCED_PARAMETER(p);
		}
	};

	template<typename T>
	class StandardAllocPolicy
	{
	public: // Typedefs
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;

	public: // Convert an StandardAllocPolicy<T> to StandardAllocPolicy<U>
		template<typename U>
		struct rebind
		{
			using other = StandardAllocPolicy<U>;
		};

	public:
		explicit StandardAllocPolicy() noexcept = default;
		explicit StandardAllocPolicy(StandardAllocPolicy const&) noexcept = default;
		template <typename U>
		explicit StandardAllocPolicy(StandardAllocPolicy<U> const&) {}

	public: // Memory allocation
		pointer allocate(size_type cnt, const_pointer = nullptr)
		{
			return reinterpret_cast<pointer>(::operator new(cnt * sizeof(T)));
		}
		void deallocate(pointer p, size_type)
		{
			::operator delete(p);
		}

	public: // Size
		size_type max_size() const
		{
			return std::numeric_limits<size_type>::max();
		}
	};

	/*
	 *	Determines if memory from another
	 *	allocator can be deallocated from this one
	 */
	template<typename T, typename T2>
	bool operator==(StandardAllocPolicy<T> const&, StandardAllocPolicy<T2> const&)
	{
		return true;
	}
	template<typename T, typename OtherAllocator>
	bool operator==(StandardAllocPolicy<T> const&, OtherAllocator const&)
	{
		return false;
	}

	template<typename T, int Alignment = 16>
	class AlignedAllocPolicy
	{
	public: // Typedefs
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;

	public: // Convert an AlignedAllocPolicy<T> to AlignedAllocPolicy<U>
		template<typename U>
		struct rebind
		{
			using other = AlignedAllocPolicy<U, Alignment>;
		};

	public:
		explicit AlignedAllocPolicy() noexcept = default;
		explicit AlignedAllocPolicy(AlignedAllocPolicy const&) noexcept = default;
		template <typename U, int AlignmentRhs>
		explicit AlignedAllocPolicy(AlignedAllocPolicy<U, AlignmentRhs> const&) {}

	public: // Memory allocation
		pointer allocate(size_type cnt, const_pointer = nullptr)
		{
#if defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64)
			return reinterpret_cast<pointer>(_mm_malloc(cnt * sizeof(T), Alignment));
#else
			return reinterpret_cast<pointer>(aligned_alloc(Alignment, cnt * sizeof(T)));
#endif
		}
		void deallocate(pointer p, size_type)
		{
#if defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64)
			_mm_free(p);
#else
			free(p);
#endif
		}

	public: // Size
		size_type max_size() const
		{
			return std::numeric_limits<size_type>::max();
		}
	};

	/*
	 *	Determines if memory from another
	 *	allocator can be deallocated from this one
	 */
	template<typename T, int Alignment, typename T2, int Alignment2>
	bool operator==(AlignedAllocPolicy<T, Alignment> const&, AlignedAllocPolicy<T2, Alignment2> const&)
	{
		return true;
	}

	template<typename T, int Alignment, typename OtherAllocator>
	bool operator==(AlignedAllocPolicy<T, Alignment> const&, OtherAllocator const&)
	{
		return false;
	}

	template<typename T, typename Policy = StandardAllocPolicy<T>, typename Traits = ObjectTraits<T>>
	class Allocator : public Policy, public Traits
	{
	private: /* Typedefs */
		using AllocationPolicy = Policy;
		using TTraits = Traits;

	public: /* Typedefs */
		using size_type = typename AllocationPolicy::size_type;
		using difference_type = typename AllocationPolicy::difference_type;
		using pointer = typename AllocationPolicy::pointer;
		using const_pointer = typename AllocationPolicy::const_pointer;
		using reference = typename AllocationPolicy::reference;
		using const_reference = typename AllocationPolicy::const_reference;
		using value_type = typename AllocationPolicy::value_type;

	public:
		template <typename U>
		struct rebind
		{
			using other = Allocator<U, typename AllocationPolicy::template rebind<U>::other, typename TTraits::template rebind<U>::other>;
		};

	public:
		explicit Allocator() noexcept = default;
		Allocator(Allocator const& rhs) noexcept : Policy(rhs), Traits(rhs) {}
		template <typename U, typename P, typename T2>
		Allocator(Allocator<U, P, T2> const& rhs) : Policy(rhs), Traits(rhs) {}
	};

	/*
	 *	Determines if memory from another
	 *	allocator can be deallocated from this one
	 */
	template<typename T, typename P, typename Tr>
	bool operator==(Allocator<T, P, Tr> const& lhs, Allocator<T, P, Tr> const& rhs)
	{
		return operator==(static_cast<P&>(lhs), static_cast<P&>(rhs));
	}
	template<typename T, typename P, typename Tr, typename T2, typename P2, typename Tr2>
	bool operator==(Allocator<T, P, Tr> const& lhs, Allocator<T2, P2, Tr2> const& rhs)
	{
		return operator==(static_cast<P&>(lhs), static_cast<P2&>(rhs));
	}
	template<typename T, typename P, typename Tr, typename OtherAllocator>
	bool operator==(Allocator<T, P, Tr> const& lhs, OtherAllocator const& rhs)
	{
		return operator==(static_cast<P&>(lhs), rhs);
	}
	template<typename T, typename P, typename Tr>
	bool operator!=(Allocator<T, P, Tr> const& lhs, Allocator<T, P, Tr> const& rhs)
	{
		return !operator==(lhs, rhs);
	}
	template<typename T, typename P, typename Tr, typename T2, typename P2, typename Tr2>
	bool operator!=(Allocator<T, P, Tr> const& lhs, Allocator<T2, P2, Tr2> const& rhs)
	{
		return !operator==(lhs, rhs);
	}
	template<typename T, typename P, typename Tr, typename OtherAllocator>
	bool operator!=(Allocator<T, P, Tr> const& lhs, OtherAllocator const& rhs)
	{
		return !operator==(lhs, rhs);
	}
}}
