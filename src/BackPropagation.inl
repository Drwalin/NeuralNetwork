
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef BACK_PROPAGATION_INL
#define BACK_PROPAGATION_INL

inline NeuralNetwork & BackPropagation::AccessMainNetwork()
{
	return this->ann;
}

inline void BackPropagation::SetLearningModifier( float value )
{
	this->learningFactorModifier = value;
}

inline void BackPropagation::SetAvailbleToChangeLearningFactorMaximallyOncePerEpochs( sizetype epochs )
{
	this->modifyLearningFactorMaximallyOncePerEpochs = epochs;
}

inline void BackPropagation::SetMinMaxWeights( float min, float max )
{
	this->minWeight = min;
	this->maxWeight = max;
}

inline float * BackPropagation::AccessGradient( sizetype layer, sizetype neuron, float * gradient )
{
	return gradient + this->ann.outputsOffsetPerLayer[ layer ] + neuron - this->ann.neuronsPerLayers[0];
}

inline float * BackPropagation::AccessDeltaWeights( sizetype layer, sizetype neuron, sizetype id, float * deltaWeights )
{
	return deltaWeights + this->ann.weightsOffsetPerLayer[layer] + ( ( this->ann.neuronsPerLayers[layer-1] + 1 ) * neuron ) + id;
}

inline void BackPropagation::CalculateOutputNeuronGradient( sizetype layer, sizetype neuron, float * desiredOutput, float * gradient, sizetype threadID )
{
	float output = this->ann.AccessOutput( layer, neuron, threadID );
	
	//without optimization:
	//*this->AccessGradient( layer, neuron ) = 2.0f * ( output - desiredOutput[neuron] ) * output * ( 1.0f - output );
	//                                        <--->
	//                                         Contained in learningFactor
	
	//with optimization:
	*this->AccessGradient( layer, neuron, gradient ) = ( output - desiredOutput[neuron] ) * output * ( 1.0f - output );
}

inline void BackPropagation::CalculateHiddenNeuronGradient( sizetype layer, sizetype neuron, float * gradient_, sizetype threadID )
{
	float * gradient = this->AccessGradient( layer, neuron, gradient_ );
	float * weight = &(this->ann.AccessWeight( layer+1, 0, neuron ));
	float * inGradient = this->AccessGradient( layer+1, 0, gradient_ );
	
	*gradient = 0.0f;
	
	sizetype i;
	sizetype iterations = this->ann.neuronsPerLayers[ layer + 1 ];
	sizetype weightModifier = 1 + this->ann.neuronsPerLayers[layer];
	
	for( i = 0; i < iterations; ++i, weight += weightModifier, ++inGradient )
		*gradient += (*weight) * (*inGradient);
	
	float output = this->ann.AccessOutput( layer, neuron, threadID );
	*gradient *= output * ( 1.0f - output );
}

inline void BackPropagation::UpdateDeltaWeight( sizetype layer, sizetype neuron, float * gradient_, float * deltaWeights_, sizetype threadID )
{
	float gradient = *this->AccessGradient( layer, neuron, gradient_ );
	float * deltaWeight = this->AccessDeltaWeights( layer, neuron, 0, deltaWeights_ );
	float * input = &(this->ann.AccessOutput( layer-1, 0, threadID ));
	float sum = 0.0f;
	
	sizetype i;
	sizetype iterations = this->ann.neuronsPerLayers[layer-1];
	
	for( i = 0; i < iterations; ++i, ++deltaWeight, ++input )
		*deltaWeight -= gradient * (*input);
	
	*deltaWeight -= gradient;
}

#endif

