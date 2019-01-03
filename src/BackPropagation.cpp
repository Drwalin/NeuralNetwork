
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef BACK_PROPAGATION_CPP
#define BACK_PROPAGATION_CPP

#include "BackPropagation.h"

#include <map>
#include <vector>
#include <fstream>

void BackPropagation::SetLearningFactor( float value )
{
	this->learningFactor = value;
}

void BackPropagation::AllocateArrays()
{
	if( this->ann.IsValid() )
	{
		this->deltaWeights = Alloc<float>( this->ann.allWeights );
		this->prevDeltaWeights = Alloc<float>( this->ann.allWeights );
		this->gradient = Alloc<float>( this->ann.allNeurons - this->ann.neuronsPerLayers[0] );
	}
}

bool BackPropagation::IsValid() const
{
	if( this->ann.IsValid() == false )
		return false;
	if( this->gradient == nullptr )
		return false;
	if( this->deltaWeights == nullptr )
		return false;
	if( this->prevDeltaWeights == nullptr )
		return false;
	return true;
}

void BackPropagation::TrainOneEpoch( const Data & data )
{
	this->MSE = 0.0f;
	if( this->IsValid() )
	{
		if( this->ann.IsDataValid( data ) )
		{
			this->ClearDeltaWeights();
			
			sizetype i;
			float currentSE;
			for( i = 0; i < data.Size(); ++i )
			{
				this->ann.Run( data[i].GetInputPointer() );
				currentSE = this->ann.GetSE( data[i].GetOutputPointer() );
				this->MSE += currentSE;
				
				this->CalculateGradient( data[i].GetOutputPointer() );
				this->UpdateDeltaWeights();
			}
			
			if( data.Size() )
			{
				this->MSE /= float( data.Size() );
				
				static sizetype learnFactoModifyCooldown = this->currentEpoch + this->modifyLearningFactorMaximallyOncePerEpochs;
				if( this->MSE >= this->prevMSE )
				{
					if( learnFactoModifyCooldown < this->currentEpoch )
					{
						this->learningFactor *= learningFactorModifier;
						learnFactoModifyCooldown = this->currentEpoch + this->modifyLearningFactorMaximallyOncePerEpochs;
					}
				}
				
				this->prevMSE = this->MSE;
				
				if( this->learningFactor < 0.01f )
					this->learningFactor = 0.01f;
				
				this->UpdateWeights( data.Size() );
			}
		}
	}
}

void BackPropagation::CalculateGradient( float * desiredOutput )
{
	if( this->IsValid() )
	{
		sizetype i, j;
		
		for( j = 0; j < this->ann.neuronsPerLayers[this->ann.layers-1]; ++j )
			this->CalculateOutputNeuronGradient( this->ann.layers-1, j, desiredOutput );
		
		for( i = this->ann.layers-2; i > 0; --i )
			for( j = 0; j < this->ann.neuronsPerLayers[i]; ++j )
				this->CalculateHiddenNeuronGradient( i, j );
	}
}

void BackPropagation::UpdateDeltaWeights()
{
	if( this->IsValid() )
	{
		sizetype i, j;
		float additionalDivider;
		float * outptr;
		for( i = 1; i < this->ann.layers; ++i )
		{
			for( j = 0; j < this->ann.neuronsPerLayers[i]; ++j )
				this->UpdateDeltaWeight( i, j );
		}
	}
}

void BackPropagation::ClearDeltaWeights()
{
	if( this->IsValid() )
	{
		float * ptr = this->deltaWeights;
		float * end = ptr + this->ann.allWeights;
		for( ; ptr < end; ++ptr )
			*ptr = 0.0f;
	}
}

void BackPropagation::UpdateWeights( sizetype numberOfTrainedDataSets )
{
	if( this->IsValid() )
	{
		float * weight = this->ann.weights;
		float * deltaWeight = this->deltaWeights;
		float * weightEnd = weight + this->ann.allWeights;
		float * prevDeltaWeight = this->prevDeltaWeights;
		for( ; weight < weightEnd; ++weight, ++deltaWeight )
		{
			float delta = this->learningFactor * (*deltaWeight) / float(numberOfTrainedDataSets);
			if( delta <= -1.0f )
				delta = -1.0f;
			else if( delta >= 1.0f )
				delta = 1.0f;
			else if( delta >= -10.0f && delta <= 10.0f )
			{
			}
			else			// correct for invalid deltaWeight
				delta = 0.0f;
			*weight += delta;
			
			if( *weight < minWeight )
				*weight = minWeight;
			else if( *weight > maxWeight )
				*weight = maxWeight;
		}
	}
}

unsigned BackPropagation::LoadTrainingData( std::istream & stream )
{
	TrainingStrategy::LoadTrainingData( stream );
	unsigned ret = this->ann.LoadFromStandardStream( stream );
	stream >> this->learningFactor;
	this->AllocateArrays();
	stream >> this->minWeight;
	stream >> this->maxWeight;
	return ret;
}

unsigned BackPropagation::SaveTrainingData( std::ostream & stream ) const
{
	TrainingStrategy::SaveTrainingData( stream );
	unsigned ret = this->ann.SaveToStandardStream( stream );
	stream << this->learningFactor << "\n";
	stream << this->minWeight << "\n";
	stream << this->maxWeight << "\n";
	return ret;
}

float BackPropagation::GetCurrentError() const
{
	return this->MSE;
}

float BackPropagation::GetMSE() const
{
	return this->MSE;
}

unsigned BackPropagation::CreateSquareErrorAnalysisPerAllDataSet( const char * fileName, const Data & data, float writeOnlyGraterThan )
{
	std::ofstream file( fileName );
	
	if( file.good() )
	{
		std::map < std::vector<long long>, std::vector<float> > dataSet;
		
		sizetype i, j;
		float currentSE;
		std::vector < long long > desiredOutputVector;
		desiredOutputVector.resize( 32 );
		for( i = 0; i < data.Size(); ++i )
		{
			this->ann.Run( data[i].GetInputPointer() );
			currentSE = this->ann.GetSE( data[i].GetOutputPointer() );
			for( j = 0; j < 32; ++j )
				desiredOutputVector[j] = (long long)( data[i].GetOutputPointer()[j] * 10000.0f );
			std::vector < float > & temp = dataSet[desiredOutputVector];
			temp.resize( temp.size() + 1 );
			temp.back() = currentSE;
		}
		
		for( auto it = dataSet.begin(); it != dataSet.end(); ++it )
		{
			file << "\nDesired output:\n   ";
			for( i = 0; i < it->first.size(); ++i )
			{
				file << " " << float(it->first[i]) / 10000.0f;
			}
			
			for( i = 0; i < it->second.size(); ++i )
			{
				if( it->second[i] > writeOnlyGraterThan )
				{
					file << "\n        " << it->second[i];
				}
			}
		}
		
		file.close();
		return 0;
	}
	
	return 1;
}

void BackPropagation::Init( sizetype layers, const sizetype * const neurons )
{
	this->ann.Init( layers, neurons );
	this->AllocateArrays();
	this->currentEpoch = 0;
}

void BackPropagation::Destroy()
{
	TrainingStrategy::Destroy();
	this->ann.Destroy();
	Free( this->gradient );
	Free( this->deltaWeights );
	Free( this->prevDeltaWeights );
	this->learningFactor = 0.01f;
	this->gradient = nullptr;
	this->deltaWeights = nullptr;
	this->prevDeltaWeights = nullptr;
	this->MSE = 0.0f;
}

BackPropagation::BackPropagation()
{
	learningFactorModifier = 0.997f;
	modifyLearningFactorMaximallyOncePerEpochs = 100;
	minWeight = -1500.0f;
	maxWeight = 1500.0f;
	learningFactor = 1.0f;
	gradient = nullptr;
	deltaWeights = nullptr;
	prevDeltaWeights = nullptr;
	MSE = 0.0f;
}

BackPropagation::~BackPropagation()
{
	Destroy();
}

#endif

