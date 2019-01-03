
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef RANDOM_CPP
#define RANDOM_CPP

#include <random>
#include <chrono>

namespace Random
{
	std::default_random_engine generator( std::chrono::system_clock::now().time_since_epoch().count() );
	std::normal_distribution < float > normal_distribution( 0.0f, 1.0f );
	std::uniform_real_distribution < float > uniform_real_distribution( -1.0f, 1.0f );
	
	float Random( float max )
	{
		float ret;
		do
		{
			ret = normal_distribution( generator );
		}
		while( ret < -max || ret > max );
		return ret;
	}
	
	float Random( float max, float deviation )
	{
		float ret;
		do
		{
			ret = normal_distribution( generator ) * deviation;
		}
		while( ret < -max || ret > max );
		return ret;
	}
	
	float UniformRandom( float deviation )
	{
		return uniform_real_distribution( generator ) * deviation;
	}
};

#endif

