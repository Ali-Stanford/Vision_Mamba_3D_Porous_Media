/*
 * File: IntStack.h
 * 
 * integer stack whose size can be dynamically doubled when
 * the stack is full. 
 *
 * Youngseuk Keehm, Nov. 29, 2001 
 */

#ifndef _IntQueue
#define _IntQueue

#include <iostream>
class IntQueue
{
public:
	IntQueue(int initSize = 100);
	~IntQueue();
	void insert(int NewValue);
	int remove();
	int size(){ return _size;};
	bool isempty() {return _size!=0;};
    
private:
	int* que;
	int  _size, _memSize;
	int  _is, _ie;
};

#endif
