
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef NEURAL_NETWORK_CPP
#define NEURAL_NETWORK_CPP

#include "NeuralNetwork.h"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <cassert>

#include <fstream>

#include "Random.h"

bool NeuralNetwork::IsValid() const
{
	if( this->weights == nullptr )
	{
		printf( "\n NeuralNetwork is not valid: weights" );
		return false;
	}
	
	if( this->outputs.size() == 0 )
	{
		printf( "\n NeuralNetwork is not valid: outputs size equal null" );
		return false;
	}
	for( int i = 0; i < this->outputs.size(); ++i )
	{
		if( this->outputs[i] == nullptr )
		{
			printf( "\n NeuralNetwork is not valid: outputs[%i]", i );
			return false;
		}
	}
	
	if( this->neuronsPerLayers == nullptr )
	{
		printf( "\n NeuralNetwork is not valid: neuronsPerLayers" );
		return false;
	}
	if( this->weightsOffsetPerLayer == nullptr )
	{
		printf( "\n NeuralNetwork is not valid: weightsOffsetPerLayer" );
		return false;
	}
	if( this->outputsOffsetPerLayer == nullptr )
	{
		printf( "\n NeuralNetwork is not valid: outputsOffsetPerLayer" );
		return false;
	}
	return true;
}

void NeuralNetwork::Run( const float * inputs, sizetype outputsID )
{
	while( this->outputs.size() <= outputsID )
	{
		this->outputs.resize( this->outputs.size() + 1 );
		this->outputs.back() = Alloc<float>( this->allNeurons );
	}
	
	float * outputs = this->outputs[outputsID];
	if( this->IsValid() && inputs )
	{
		sizetype i, j;
		for( i = 0; i < *this->neuronsPerLayers; ++i )
			outputs[i] = inputs[i];
		
		for( i = 1; i < this->layers; ++i )
			for( j = 0; j < this->neuronsPerLayers[i]; ++j )
				this->CalculateNeuronOutput( i, j, outputsID );
	}
}

float * NeuralNetwork::GetOutputs( sizetype outputsID )
{
	if( this->IsValid() )
		return this->outputs[outputsID] + this->outputsOffsetPerLayer[ this->layers-1 ];
	return nullptr;
}

void NeuralNetwork::Randomize( float max, float deviation )
{
	if( this->IsValid() )
	{
		sizetype i;
		for( i = 0; i < this->allWeights; ++i )
			this->weights[i] += Random::Random( max, deviation );
	}
}

void NeuralNetwork::InitRandom( float max, float deviation )
{
	if( this->IsValid() )
	{
		sizetype i, j, k, w = 0;
		for( i = 1; i < this->layers; ++i )
		{
			float multiplier = sqrt( 2.0f / float( this->neuronsPerLayers[i-1] ) );
			for( j = 0; j < this->neuronsPerLayers[i]; ++j )
			{
				for( k = 0; k < this->neuronsPerLayers[i-1]; ++k, ++w )
				{
					this->weights[w] = Random::Random( max, deviation ) * multiplier;
				}
			}
		}
	}
}

unsigned NeuralNetwork::Init( sizetype layers, const sizetype * const neurons )
{
	this->Destroy();
	
	if( layers && neurons )
	{
		sizetype i, j, k;
		this->allWeights = 0;
		this->allNeurons = 0;		// included inputs, hidden and outputs
		this->layers = layers;
		this->neuronsPerLayers = Alloc<sizetype>( layers );
		assert( this->neuronsPerLayers );
		
		for( i = 0; i < layers; ++i )
		{
			this->neuronsPerLayers[i] = neurons[i];
			allNeurons += neurons[i];
			if( i > 0 )
				allWeights += this->neuronsPerLayers[i] * ( this->neuronsPerLayers[i-1] + 1 );		// "+1" - bias
		}
		
		this->weights = Alloc<float>( this->allWeights );
		this->outputs.resize( 1 );
		this->outputs.back() = Alloc<float>( this->allNeurons );
		assert( this->weights );
		assert( this->outputs.back() );
		memset( this->weights, 0, this->allWeights * sizeof( float ) );
		memset( this->outputs.back(), 0, this->allNeurons * sizeof( float ) );
		
		this->weightsOffsetPerLayer = Alloc<sizetype>( layers );
		this->outputsOffsetPerLayer = Alloc<sizetype>( layers );
		assert( this->weightsOffsetPerLayer );
		assert( this->outputsOffsetPerLayer );
		
		this->weightsOffsetPerLayer[0] = 0;
		this->outputsOffsetPerLayer[0] = 0;
		
		sizetype currentOutputsOffset = this->neuronsPerLayers[0];
		sizetype currentWeightsOffset = 0;
		
		for( i = 1; i < layers; ++i )
		{
			this->outputsOffsetPerLayer[i] = currentOutputsOffset;
			this->weightsOffsetPerLayer[i] = currentWeightsOffset;
			
			currentOutputsOffset += this->neuronsPerLayers[i];
			currentWeightsOffset += this->neuronsPerLayers[i] * ( this->neuronsPerLayers[i-1] + 1 );		// "+1" - bias
		}
		
		return 0;
	}
	else
	{
		printf( "\n Invalid initiation of neural network" );
	}
	
	return 1;
}

unsigned NeuralNetwork::LoadFromFile( const char * fileName )
{
	if( fileName )
	{
		std::ifstream file( fileName );
		if( file.good() )
		{
			sizetype ret = this->LoadFromStandardStream( file );
			file.close();
			return ret;
		}
		else
		{
			return 2;
		}
	}
	return 1;
}

unsigned NeuralNetwork::SaveToFile( const char * fileName ) const
{
	if( fileName )
	{
		std::ofstream file( fileName );
		if( file.good() )
		{
			sizetype ret = this->SaveToStandardStream( file );
			file.close();
			return ret;
		}
		else
		{
			return 2;
		}
	}
	return 1;
}

unsigned NeuralNetwork::LoadFromStandardStream( std::istream & stream )
{
	this->Destroy();
	
	if( !stream.good() )
		return 3;
	
	stream >> this->layers;
	stream >> this->allWeights;
	stream >> this->allNeurons;
	
	if( stream.fail() || stream.eof() ) { this->Destroy(); return 4; }
	
	this->neuronsPerLayers = Alloc<sizetype>( this->layers );
	assert( this->neuronsPerLayers );
	
	this->outputs.resize( 1 );
	
	this->weights = Alloc<float>( this->allWeights );
	this->outputs.back() = Alloc<float>( this->allNeurons );
	assert( this->weights );
	assert( this->outputs.back() );
	memset( this->outputs.back(), 0, this->allNeurons * sizeof( float ) );
	
	this->weightsOffsetPerLayer = Alloc<sizetype>( this->layers );
	this->outputsOffsetPerLayer = Alloc<sizetype>( this->layers );
	assert( this->weightsOffsetPerLayer );
	assert( this->outputsOffsetPerLayer );
	
	sizetype i;
	
	for( i = 0; i < this->layers; ++i )
		stream >> this->neuronsPerLayers[i];
	if( stream.fail() || stream.eof() ) { this->Destroy(); return 5; }
	
	for( i = 0; i < this->allWeights; ++i )
		stream >> this->weights[i];
	if( stream.fail() || stream.eof() ) { this->Destroy(); return 6; }
	
	for( i = 0; i < this->layers; ++i )
		stream >> this->weightsOffsetPerLayer[i];
	if( stream.fail() || stream.eof() ) { this->Destroy(); return 7; }
	
	for( i = 0; i < this->layers; ++i )
		stream >> this->outputsOffsetPerLayer[i];
	
	if( stream.fail() ) return 8;
	
	return 0;
}

unsigned NeuralNetwork::SaveToStandardStream( std::ostream & stream ) const
{
	if( !stream.good() )
		return 3;
	
	stream << this->layers << "\n";
	stream << this->allWeights << "\n";
	stream << this->allNeurons << "\n";
	
	if( stream.fail() ) { return 4; }
	
	sizetype i;
	
	for( i = 0; i < this->layers; ++i )
		stream << this->neuronsPerLayers[i] << " ";
	stream << "\n";
	if( stream.fail() ) { return 5; }
	
	for( i = 0; i < this->allWeights; ++i )
		stream << this->weights[i] << " ";
	stream << "\n";
	if( stream.fail() ) { return 6; }
	
	for( i = 0; i < this->layers; ++i )
		stream << this->weightsOffsetPerLayer[i] << " ";
	stream << "\n";
	if( stream.fail() ) { return 7; }
	
	for( i = 0; i < this->layers; ++i )
		stream << this->outputsOffsetPerLayer[i] << " ";
	stream << "\n";
	
	if( stream.fail() ) return 8;
	
	return 0;
}

void NeuralNetwork::Destroy()
{
	Free( this->weights );
	for( int i = 0; i < this->outputs.size(); ++i )
		Free( this->outputs[i] );
	
	Free( this->neuronsPerLayers );
	Free( this->weightsOffsetPerLayer );
	Free( this->outputsOffsetPerLayer );
	
	this->weights = nullptr;
	this->outputs.clear();
	
	this->neuronsPerLayers = nullptr;
	this->layers = 0;
	this->weightsOffsetPerLayer = nullptr;
	this->outputsOffsetPerLayer = nullptr;
	
	this->allWeights = 0;
	this->allNeurons = 0;
}

NeuralNetwork::NeuralNetwork()
{
	this->weights = nullptr;
	
	this->neuronsPerLayers = nullptr;
	this->layers = 0;
	this->weightsOffsetPerLayer = nullptr;
	this->outputsOffsetPerLayer = nullptr;
	
	this->allWeights = 0;
	this->allNeurons = 0;
}

NeuralNetwork::~NeuralNetwork()
{
	this->Destroy();
}

#endif

