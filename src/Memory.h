
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef MEMORY_H
#define MEMORY_H

#include <cstdlib>
#include <set>

typedef unsigned long long int sizetype;

template < typename T >
inline T * Alloc( sizetype count )
{
	T * temp = (T*)malloc( count * sizeof(T) );
	return temp;
}

inline void Free( void * ptr )
{
	if( ptr )
	{
		free( ptr );
	}
}

#endif

