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
#ifndef VCL_ABI_WINAPI
#include <mm_malloc.h> // Required for _mm_malloc
#endif
#include <stddef.h>    // Required for size_t and ptrdiff_t and NULL
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
		typedef T value_type;
		typedef value_type* pointer;

	public:
		//! Convert an ObjectTraits<T> to ObjectTraits<U>
		template<typename U>
		struct rebind
		{
			typedef ObjectTraits<U> other;
		};

	public:
		//! Default constructor
		explicit ObjectTraits() = default;

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
		typedef T value_type;
		typedef value_type* pointer;

	public:
		//! Convert an NoInitObjectTraits<T> to NoInitObjectTraits<U>
		template<typename U>
		struct rebind
		{
			typedef NoInitObjectTraits<U> other;
		};

	public:
		//! Default constructor
		explicit NoInitObjectTraits() = default;

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
		typedef T value_type;
		typedef value_type* pointer;
		typedef const value_type* const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

	public: // Convert an StandardAllocPolicy<T> to StandardAllocPolicy<U>
		template<typename U>
		struct rebind
		{
			typedef StandardAllocPolicy<U> other;
		};

	public:
		inline explicit StandardAllocPolicy() {}
		inline ~StandardAllocPolicy() {}
		inline explicit StandardAllocPolicy(StandardAllocPolicy const&) {}
		template <typename U>
		inline explicit StandardAllocPolicy(StandardAllocPolicy<U> const&) {}

	public: // Memory allocation
		inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = nullptr)
		{
			return reinterpret_cast<pointer>(::operator new(cnt * sizeof(T)));
		}
		inline void deallocate(pointer p, size_type)
		{
			::operator delete(p);
		}

	public: // Size
		inline size_type max_size() const
		{
			return std::numeric_limits<size_type>::max();
		}
	};

	/*
	 *	Determines if memory from another
	 *	allocator can be deallocated from this one
	 */
	template<typename T, typename T2>
	inline bool operator==(StandardAllocPolicy<T> const&, StandardAllocPolicy<T2> const&)
	{
		return true;
	}
	template<typename T, typename OtherAllocator>
	inline bool operator==(StandardAllocPolicy<T> const&, OtherAllocator const&)
	{
		return false;
	}

	template<typename T, int Alignment = 16>
	class AlignedAllocPolicy
	{
	public: // Typedefs
		typedef T value_type;
		typedef value_type* pointer;
		typedef const value_type* const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

	public: // Convert an AlignedAllocPolicy<T> to AlignedAllocPolicy<U>
		template<typename U>
		struct rebind
		{
			typedef AlignedAllocPolicy<U, Alignment> other;
		};

	public:
		inline explicit AlignedAllocPolicy() {}
		inline ~AlignedAllocPolicy() {}
		inline explicit AlignedAllocPolicy(AlignedAllocPolicy const&) {}
		template <typename U, int AlignmentRhs>
		inline explicit AlignedAllocPolicy(AlignedAllocPolicy<U, AlignmentRhs> const&) {}

	public: // Memory allocation
		inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = nullptr)
		{
			return reinterpret_cast<pointer>(_mm_malloc(cnt * sizeof(T), Alignment));
		}
		inline void deallocate(pointer p, size_type)
		{
			_mm_free(p);
		}

	public: // Size
		inline size_type max_size() const
		{
			return std::numeric_limits<size_type>::max();
		}
	};

	/*
	 *	Determines if memory from another
	 *	allocator can be deallocated from this one
	 */
	template<typename T, int Alignment, typename T2, int Alignment2>
	inline bool operator==(AlignedAllocPolicy<T, Alignment> const&, AlignedAllocPolicy<T2, Alignment2> const&)
	{
		return true;
	}

	template<typename T, int Alignment, typename OtherAllocator>
	inline bool operator==(AlignedAllocPolicy<T, Alignment> const&, OtherAllocator const&)
	{
		return false;
	}

	template<typename T, typename Policy = StandardAllocPolicy<T>, typename Traits = ObjectTraits<T>>
	class Allocator : public Policy, public Traits
	{
	private: /* Typedefs */
		typedef Policy AllocationPolicy;
		typedef Traits TTraits;

	public: /* Typedefs */
		typedef typename AllocationPolicy::size_type size_type;
		typedef typename AllocationPolicy::difference_type difference_type;
		typedef typename AllocationPolicy::pointer pointer;
		typedef typename AllocationPolicy::const_pointer const_pointer;
		typedef typename AllocationPolicy::reference reference;
		typedef typename AllocationPolicy::const_reference const_reference;
		typedef typename AllocationPolicy::value_type value_type;

	public:
		template <typename U>
		struct rebind
		{
			typedef Allocator<U, typename AllocationPolicy::template rebind<U>::other, typename TTraits::template rebind<U>::other> other;
		};

	public:
		inline explicit Allocator() {}
		inline ~Allocator() {}
		inline Allocator(Allocator const& rhs) : Policy(rhs), Traits(rhs) {}
		template <typename U, typename P, typename T2>
		inline Allocator(Allocator<U, P, T2> const& rhs) : Policy(rhs), Traits(rhs) {}
	};

	/*
	 *	Determines if memory from another
	 *	allocator can be deallocated from this one
	 */
	template<typename T, typename P, typename Tr>
	inline bool operator==(Allocator<T, P, Tr> const& lhs, Allocator<T, P, Tr> const& rhs)
	{
		return operator==(static_cast<P&>(lhs), static_cast<P&>(rhs));
	}
	template<typename T, typename P, typename Tr, typename T2, typename P2, typename Tr2>
	inline bool operator==(Allocator<T, P, Tr> const& lhs, Allocator<T2, P2, Tr2> const& rhs)
	{
		return operator==(static_cast<P&>(lhs), static_cast<P2&>(rhs));
	}
	template<typename T, typename P, typename Tr, typename OtherAllocator>
	inline bool operator==(Allocator<T, P, Tr> const& lhs, OtherAllocator const& rhs)
	{
		return operator==(static_cast<P&>(lhs), rhs);
	}
	template<typename T, typename P, typename Tr>
	inline bool operator!=(Allocator<T, P, Tr> const& lhs, Allocator<T, P, Tr> const& rhs)
	{
		return !operator==(lhs, rhs);
	}
	template<typename T, typename P, typename Tr, typename T2, typename P2, typename Tr2>
	inline bool operator!=(Allocator<T, P, Tr> const& lhs, Allocator<T2, P2, Tr2> const& rhs)
	{
		return !operator==(lhs, rhs);
	}
	template<typename T, typename P, typename Tr, typename OtherAllocator>
	inline bool operator!=(Allocator<T, P, Tr> const& lhs, OtherAllocator const& rhs)
	{
		return !operator==(lhs, rhs);
	}
}}
