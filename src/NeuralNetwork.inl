
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef NEURAL_NETWORK_INL
#define NEURAL_NETWORK_INL

#include <cmath>

#include "Random.h"

inline void NeuralNetwork::ClampWeights( float min, float max )
{
	float * weight = this->weights;
	float * end = weight + this->allWeights;
	for( ; weight < end; ++weight )
	{
		if( *weight >= min && *weight <= max )
		{
			continue;
		}
		else if( *weight <= min )
			*weight = min;
		else if( *weight >= max )
			*weight = max;
		else
		{
			*weight = Random::Random( 10.0f );
			--weight;
		}
	}
}

inline bool NeuralNetwork::IsDataSetValid( const DataSet & dataSet ) const
{
	if( this->neuronsPerLayers )
		if( this->neuronsPerLayers[0] == dataSet.GetInputs() && this->neuronsPerLayers[this->layers-1] == dataSet.GetOutputs() )
			return true;
	return false;
}

inline bool NeuralNetwork::IsDataValid( const Data & data ) const
{
	if( this->neuronsPerLayers )
		if( this->neuronsPerLayers[0] == data.GetInputs() && this->neuronsPerLayers[this->layers-1] == data.GetOutputs() )
			return true;
	return false;
}

inline sizetype NeuralNetwork::GetNumberOfInputs() const
{
	if( this->IsValid() )
		return this->neuronsPerLayers[0];
	return 0;
}

inline sizetype NeuralNetwork::GetNumberOfOutputs() const
{
	if( this->IsValid() )
		return this->neuronsPerLayers[this->layers-1];
	return 0;
}

inline float NeuralNetwork::ActivationFunction( float sum )
{
	float val = 1.0f / ( 1.0f + exp( -sum ) );
	if( !(val < 2.0f ) )
		printf( "\n function sum = %6.6f val = %6.6f ", sum, val );
	return val;
	
	
	if( sum >= -6.0f && sum <= 6.0f )
	{
		float val = 1.0f / ( 1.0f + exp( -sum ) );
		if( !(val < 2.0f ) )
			printf( "\n function sum = %6.6f val = %6.6f ", sum, val );
		return val;
	}
	else if( sum <= -5.9999f )
	{
		return 0.00025f;
	}
	else if( sum >= 5.9999f )
	{
		return 0.99975f;
	}
	else
	{
		printf( "\n Error while calculating activation function. sum = %f ", sum );
	}
	return 0.000001f;
}

inline float NeuralNetwork::ActivationFunctionDerivative( float sum )
{
	if( sum >= -6.0f && sum <= 6.0f )
	{
		float e = exp( -sum );
		float d = ( 1.0f + e );
		float val = e / ( d * d );
		if( !(val < 200.0f ) )
			printf( "\n derivative sum = %6.6f val = %6.6f ", sum, val );
		return val;
	}
	else if( sum <= -5.9999f )
	{
		return 0.00025f;
	}
	else if( sum >= 5.9999f )
	{
		return 0.00025f;
	}
	else
	{
		printf( "\n Error while calculating activation function derivative. sum = %f ", sum );
	}
	return 0.000001f;
}

inline void NeuralNetwork::CalculateNeuronOutput( sizetype layerId, sizetype id )		// id - neuronID
{
	float sum = 0.0f;
	float * weight = this->weights + this->weightsOffsetPerLayer[layerId] + ( ( this->neuronsPerLayers[layerId-1] + 1 ) * id );
	float * input = this->outputs + this->outputsOffsetPerLayer[layerId-1];
	float * endWeights = weight + this->neuronsPerLayers[layerId-1];
	for( ; weight < endWeights; ++weight, ++input )
	{
		sum += (*weight) * (*input);
	}
	sum += *weight;
	this->outputs[ this->outputsOffsetPerLayer[layerId] + id ] = NeuralNetwork::ActivationFunction( sum );
}

inline float& NeuralNetwork::AccessWeight( sizetype layer, sizetype neuron, sizetype weightID )
{
#ifdef CHECK_VALIDITY
	if( this->IsValid() && layer < this->layers && layer && neuron < this->neuronsPerLayers[layer] )
#endif
	{
		return this->weights[ this->weightsOffsetPerLayer[layer] + ( ( this->neuronsPerLayers[layer-1] + 1 ) * neuron ) + weightID ];
	}
	
	NeuralNetwork::FREE_ACCESS_FOR_INVALID_NETWORKS = 0.0f;
	return NeuralNetwork::FREE_ACCESS_FOR_INVALID_NETWORKS;
}

inline float& NeuralNetwork::AccessOutput( sizetype layer, sizetype neuron )
{
#ifdef CHECK_VALIDITY
	if( this->IsValid() && layer < this->layers && layer && neuron < this->neuronsPerLayers[layer] )
#endif
	{
		return this->outputs[ this->outputsOffsetPerLayer[layer] + neuron ];
	}
	
	printf( "\n return NeuralNetwork::FREE_ACCESS_FOR_INVALID_NETWORKS;" );
	NeuralNetwork::FREE_ACCESS_FOR_INVALID_NETWORKS = 0.0f;
	return NeuralNetwork::FREE_ACCESS_FOR_INVALID_NETWORKS;
}

inline float NeuralNetwork::GetSE( const float * desiredOutput ) const
{
	if( this->IsValid() && desiredOutput && this->neuronsPerLayers[this->layers-1] )
	{
		float ret = 0.0f;
		float * dst = (float*)desiredOutput;
		float * out = this->outputs + this->outputsOffsetPerLayer[this->layers-1];
		float * endOut = out + this->neuronsPerLayers[this->layers-1];
		for( ; out < endOut; ++out, ++dst )
		{
			float temp = (*out) - (*dst);
			ret += temp * temp;
		}
		return ret;
	}
	
	return -1.0f;
}

#endif

