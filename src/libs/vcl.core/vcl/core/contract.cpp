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
#include <vcl/core/contract.h>

// C++ standard library
#include <iostream>

// C runtime library
#include <stdarg.h>

#ifdef VCL_CONTRACT
namespace Vcl { namespace Assert
{
	enum class QueryAnswer
	{
		IgnoreOnce = 1,
		IgnoreForEver = 2,
		Debug = 3
	};

#ifdef VCL_ABI_WINAPI

	#include <windows.h>

	// Register a windows hook to change the message box layout
	thread_local HHOOK hhk = nullptr;

	LRESULT CALLBACK CBTProc(INT nCode, WPARAM wParam, LPARAM lParam)
	{
	   HWND hChildWnd;    // msgbox is "child"
	   // notification that a window is about to be activated
	   // window handle is wParam
	   if (nCode == HCBT_ACTIVATE)
	   {
		  // set window handles
		  hChildWnd = (HWND)wParam;
		  //to get the text of the Yes button
		  UINT result;
		  if (GetDlgItem(hChildWnd,IDIGNORE)!=NULL)
		  {         
			 result= SetDlgItemTextA(hChildWnd,IDIGNORE,"Ignore forever");
		  }
		  if (GetDlgItem(hChildWnd,IDRETRY)!=NULL)
		  {
			 
			 result= SetDlgItemTextA(hChildWnd,IDRETRY,"Ignore once");
		  }
		  if (GetDlgItem(hChildWnd,IDABORT)!=NULL)
		  {
			 
			 result= SetDlgItemTextA(hChildWnd,IDABORT,"Debug");
		  }

		  // exit CBT hook
		  UnhookWindowsHookEx(hhk);
	   }
	   // otherwise, continue with any possible chained hooks
	   else
	   {
		   CallNextHookEx(hhk, nCode, wParam, lParam);
	   }
	   return 0;
	}

	INT CBTMessageBox(HWND hwnd, const char* message, const char* title, UINT type)
	{
		hhk = SetWindowsHookEx(WH_CBT, &CBTProc, 0, GetCurrentThreadId());
		return MessageBoxA(hwnd, message, title, type);
	}

	QueryAnswer queryUser(const char* title, const char* message)
	{
		int result = CBTMessageBox(nullptr, message, title, MB_ABORTRETRYIGNORE);

		if (result == IDIGNORE)
		{
			return QueryAnswer::IgnoreForEver;
		}
		else if (result == IDRETRY)
		{
			return QueryAnswer::IgnoreOnce;
		}
		else if (result == IDABORT)
		{
			return QueryAnswer::Debug;
		}
		else
		{
			return QueryAnswer::IgnoreOnce;
		}
	}
#else

	QueryAnswer queryUser(const char* title, const char* message)
	{
		using namespace std;

		char answer = 0;
		cout << title << endl << message << endl << "(i)gnore for ever | ignore (o)nce | (d)ebug? ";

		while (true) {
			cin >> answer;
			switch (answer)
			{
			case 'i':
				return QueryAnswer::IgnoreForEver;
			case 'o':
				return QueryAnswer::IgnoreOnce;
			case 'd':
				return QueryAnswer::Debug;
			}
			cout << endl << "Unexpected input('"<< answer << "')! Choose either (i)gnore | (d)ebug: ";
		}
	}

#endif
	
	bool handler(const char* title, const char* message, bool* b)
	{
		QueryAnswer result = queryUser(title, message);

		switch (result)
		{
		case QueryAnswer::IgnoreForEver:
			*b = true;
			return false;

		case QueryAnswer::IgnoreOnce:
			*b = false;
			return false;

		case QueryAnswer::Debug:
			*b = false;
			return true;
		}

		*b = false;
		return false;
	}
}}

#endif
