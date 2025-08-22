#include "IntQueue.h"

// Constructor 
IntQueue::IntQueue(int initSize)
{
   	que	 = new int[initSize];
   	_size	 = 0;
	_memSize = initSize;
	_is		 = 0;
	_ie		 = 0;
}

// Destructor 
IntQueue::~IntQueue()
{
	delete [] que;
}
 
// insert() - add an elements at the end of entry
void IntQueue::insert(int newValue)
{
	if( _size >= _memSize  ){
		int i;
		int* tmp = que;
		que = new int[_memSize*2];
		if( _is != _ie ){
			std::cout << " Shit\n";
		}
			for(i=_is ; i<_memSize ; i++)
				que[i-_is] = tmp[i];
			for(i=0 ; i<_ie ; i++)
				que[_memSize-_is+i] = tmp[i];

		delete [] tmp;
		_memSize *= 2;
		_is = 0;
		_ie = _size;
	}
	if(_ie+1 >= _memSize){
		que[_ie] = newValue;
		_ie = 0;
	}
	else que[_ie++] = newValue;
	_size++;
}

// remove() - return and remove the first of entry
int IntQueue::remove()
{
	if(_size == 0) return 0;

	int tmp = que[_is];
	_size--;
	if(_is+1 >= _memSize) _is = 0;
	else _is++;

	return tmp;
}
