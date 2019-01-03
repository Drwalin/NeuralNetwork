
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>

#include "DataSet.h"
#include "Data.h"
#include "Memory.h"

class NeuralNetwork
{
protected:
	
	static float FREE_ACCESS_FOR_INVALID_NETWORKS;
	
	float * weights;
	float * outputs;
	
	sizetype * neuronsPerLayers;		// [0] - number of inputs ; [layers-1] - number of outputs
	sizetype layers;
	sizetype * weightsOffsetPerLayer;
	sizetype * outputsOffsetPerLayer;
	
	sizetype allWeights;		// included biases
	sizetype allNeurons;		// included inputs, hidden and outputs
	
	
	friend class LearnMethod;
	friend class BackPropagation;
	
public:
	
	inline void ClampWeights( float min, float max );
	
	inline bool IsDataSetValid( const DataSet & dataSet ) const;
	inline bool IsDataValid( const Data & data ) const;
	
	inline sizetype GetNumberOfInputs() const;
	inline sizetype GetNumberOfOutputs() const;
	
	bool IsValid() const;
	
	static inline float ActivationFunction( float sum );
	static inline float ActivationFunctionDerivative( float sum );
	inline void CalculateNeuronOutput( sizetype layerId, sizetype id );
	
	void Run( const float * inputs );
	float * GetOutputs();
	inline float GetSE( const float * desiredOutput ) const;		// return square error of output
	
	// weight( layer, neuron, neuronsPerLayers[layer-1] ) = bias
	inline float& AccessWeight( sizetype layer, sizetype neuron, sizetype weightID );
	inline float& AccessOutput( sizetype layer, sizetype neuron );
	
	void Randomize( float max, float deviation );
	void InitRandom( float max = 30.0f, float deviation = 1.0f );
	
	unsigned Init( sizetype layers, const sizetype * const neurons );
	unsigned LoadFromFile( const char * fileName );
	unsigned SaveToFile( const char * fileName ) const;
	unsigned LoadFromStandardStream( std::istream & stream );
	unsigned SaveToStandardStream( std::ostream & stream ) const;
	
	void Destroy();
	
	NeuralNetwork();
	~NeuralNetwork();
};

#include "NeuralNetwork.inl"

#endif

