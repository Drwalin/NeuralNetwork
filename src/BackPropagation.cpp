
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

float BackPropagation::GetLearningFactor() const
{
	return this->learningFactor;
}

void BackPropagation::SetLearningFactor( float value )
{
	this->learningFactor = value;
}

void BackPropagation::AllocateArrays( sizetype arraysCount )
{
	for( int i = 0; i < this->deltaWeights.size(); ++i )
	{
		if( this->deltaWeights[i] );
			Free( this->deltaWeights[i] );
	}
	this->deltaWeights.clear();
	
	for( int i = 0; i < this->gradient.size(); ++i )
	{
		if( this->gradient[i] );
			Free( this->gradient[i] );
	}
	this->gradient.clear();
	
	if( this->prevDeltaWeights )
		Free( this->prevDeltaWeights );
	this->prevDeltaWeights = nullptr;
	
	if( this->ann.IsValid() )
	{
		this->deltaWeights.resize( arraysCount );
		for( int i = 0; i < this->deltaWeights.size(); ++i )
			this->deltaWeights[i] = Alloc<float>( this->ann.allWeights );
			
		this->gradient.resize( arraysCount );
		for( int i = 0; i < this->gradient.size(); ++i )
			this->gradient[i] = Alloc<float>( this->ann.allNeurons - this->ann.neuronsPerLayers[0] );
		
		this->prevDeltaWeights = Alloc<float>( this->ann.allWeights );
	}
}

bool BackPropagation::IsValid() const
{
	if( this->ann.IsValid() == false )
		return false;
	
	if( this->gradient.size() == 0 )
		return false;
	for( int i = 0; i < this->gradient.size(); ++i )
		if( this->gradient[i] == nullptr )
			return false;
	
	if( this->deltaWeights.size() == 0 )
		return false;
	for( int i = 0; i < this->deltaWeights.size(); ++i )
		if( this->deltaWeights[i] == nullptr )
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
			// begin threads
			for( sizetype i = 0; i < this->threadsInfo.size(); ++i )
				this->threadsInfo[i]->data.store( (void*)(&data) );
			for( sizetype i = 1; i < this->threadsInfo.size(); ++i )
				this->threadsInfo[i]->flags.fetch_or( 1<<0 );
			
			this->ThreadFunction( 0 );
			this->MSE = this->threadsInfo[0]->MSE;
			
			// end threads
			for( sizetype i = 1; i < this->threadsInfo.size(); ++i )
			{
				while( ( this->threadsInfo[i]->flags.load() & (1<<1) == 0 ) || ( this->threadsInfo[i]->flags.load() & (1<<0) == 1 ) )
					std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
				
				this->threadsInfo[i]->flags.store( 0 );
				this->MSE += this->threadsInfo[i]->MSE;
				float * src = this->deltaWeights[i];
				float * dst = this->deltaWeights[0];
				for( sizetype j = 0; j < this->ann.allWeights; ++j, ++src, ++dst )
					*dst += *src;
			}
			
			if( data.Size() )
			{
				this->MSE /= float( data.Size() );
				
				static sizetype learnFactorModifyCooldown = this->currentEpoch + this->modifyLearningFactorMaximallyOncePerEpochs;
				if( this->MSE >= this->prevMSE )
				{
					if( learnFactorModifyCooldown < this->currentEpoch )
					{
						this->learningFactor *= learningFactorModifier;
						learnFactorModifyCooldown = this->currentEpoch + this->modifyLearningFactorMaximallyOncePerEpochs;
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

void BackPropagation::CalculateGradient( float * desiredOutput, sizetype threadID )
{
	if( this->IsValid() )
	{
		sizetype i, j;
		
		float * gradient = this->gradient[threadID];
		
		for( j = 0; j < this->ann.neuronsPerLayers[this->ann.layers-1]; ++j )
			this->CalculateOutputNeuronGradient( this->ann.layers-1, j, desiredOutput, gradient, threadID );
		
		for( i = this->ann.layers-2; i > 0; --i )
			for( j = 0; j < this->ann.neuronsPerLayers[i]; ++j )
				this->CalculateHiddenNeuronGradient( i, j, gradient, threadID );
	}
}

void BackPropagation::UpdateDeltaWeights( float * gradient_, float * deltaWeights_, sizetype threadID )
{
	if( this->IsValid() )
	{
		sizetype i, j;
		for( i = 1; i < this->ann.layers; ++i )
		{
			for( j = 0; j < this->ann.neuronsPerLayers[i]; ++j )
				this->UpdateDeltaWeight( i, j, gradient_, deltaWeights_, threadID );
		}
	}
}

void BackPropagation::ClearDeltaWeights( sizetype threadID )
{
	if( this->IsValid() )
	{
		float * ptr = this->deltaWeights[threadID];
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
		float * deltaWeights = this->deltaWeights[0];
		for( int i = 0; i < this->ann.allWeights; ++i, ++weight, ++deltaWeights )
		{
			float delta = this->learningFactor * (*deltaWeights) / float(numberOfTrainedDataSets);
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
	this->AllocateArrays( this->threadsCount );
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
			this->ann.Run( data[i].GetInputPointer(), 0 );
			currentSE = this->ann.GetSE( data[i].GetOutputPointer(), 0 );
			for( j = 0; j < 32; ++j )
				desiredOutputVector[j] = (long long)( data[i].GetOutputPointer()[j] * 10000.0f );
			std::vector < float > & temp = dataSet[desiredOutputVector];
			temp.resize( temp.size() + 1 );
			temp.back() = currentSE;
		}
		
		unsigned long long GOOD = 0, ALL = 0;
		
		for( auto it = dataSet.begin(); it != dataSet.end(); ++it )
		{
			file << "\nDesired output:\n   ";
			for( i = 0; i < it->first.size(); ++i )
			{
				file << " " << float(it->first[i]) / 10000.0f;
			}
			
			unsigned long long good = 0, bad = 0;
			
			for( i = 0; i < it->second.size(); ++i )
			{
				if( it->second[i] > writeOnlyGraterThan )
				{
					++bad;
				}
				else
				{
					++good;
					++GOOD;
				}
				++ALL;
			}
			
			printf( "\n    good / all   :   %llu / %llu  =  %f%% good", good, (good+bad), (float(good)/float(good+bad))*100.0f );
			
			file << "\n    good / all   :   " << good << " / " << (good+bad) << "  =  " << (float(good)/float(good+bad))*100.0f << "% good";
			
			for( i = 0; i < it->second.size(); ++i )
			{
				if( it->second[i] > writeOnlyGraterThan )
				{
					file << "\n        " << it->second[i];
				}
			}
		}
		
		printf( "\n    Total goodness: %llu / %llu  =  %f%%", GOOD, ALL, float(GOOD)*100.0f/float(ALL) );
		
		file.close();
		return 0;
	}
	
	return 1;
}

void BackPropagation::Init( sizetype layers, const sizetype * const neurons )
{
	this->ann.Init( layers, neurons );
	this->AllocateArrays( this->threadsCount );
	this->currentEpoch = 0;
}

void BackPropagation::PreInit( sizetype threadsCount )
{
	this->threadsCount = threadsCount;
	
	this->threads.resize( this->threadsCount );
	this->threadsInfo.resize( this->threadsCount );
	
	for( sizetype i = 0; i < this->threadsCount; ++i )
	{
		this->threadsInfo[i] = new BackPropagationThreadInfo;
		this->threadsInfo[i]->threadsCount = this->threadsCount;
		this->threadsInfo[i]->threadID = i;
		this->threadsInfo[i]->backPropagation = this;
		this->threadsInfo[i]->flags.store( 0 );
		this->threadsInfo[i]->data.store( NULL );
		
		this->threads[i] = NULL;
	}
	
	for( sizetype i = 0; i < this->threadsCount; ++i )
	{
		if( i != 0 )
		{
			this->threads[i] = new std::thread( BackPropagationThreadFunction, this->threadsInfo[i] );
			this->threads[i]->detach();
		}
	}
}

void BackPropagation::ThreadFunction( sizetype threadID )
{
	this->ClearDeltaWeights( threadID );
	
	this->threadsInfo[threadID]->MSE = 0.0f;
	
	float * gradient_ = this->gradient[threadID];
	float * deltaWeights_ = this->deltaWeights[threadID];
	
	const Data * data = (const Data*)(this->threadsInfo[threadID]->data.load());
	
	sizetype i;
	float currentSE;
	for( i = threadID; i < data->Size(); i += this->threadsCount )
	{
		this->ann.Run( data->operator[](i).GetInputPointer(), threadID );
		currentSE = this->ann.GetSE( data->operator[](i).GetOutputPointer(), threadID );
		this->threadsInfo[threadID]->MSE += currentSE;
		
		this->CalculateGradient( data->operator[](i).GetOutputPointer(), threadID );
		this->UpdateDeltaWeights( gradient_, deltaWeights_, threadID );
	}
	
	this->threadsInfo[threadID]->data.store( NULL );
}

void BackPropagationThreadFunction( BackPropagation::BackPropagationThreadInfo * threadInfo )
{
	while( true )
	{
		if( threadInfo->flags.load() & (1<<2) )
		{
			break;
		}
		else if( threadInfo->flags.load() & (1<<0) )
		{
			threadInfo->backPropagation->ThreadFunction( threadInfo->threadID );
			threadInfo->flags.fetch_and( ~((unsigned)(1<<0)) );
			threadInfo->flags.fetch_or( 1<<1 );
		}
		
		std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
	}
	threadInfo->flags.fetch_or( 1<<3 );
}

void BackPropagation::Destroy()
{
	// finish threads
	{
		for( int i = 1; i < this->threadsInfo.size(); ++i )
			this->threadsInfo[i]->flags.fetch_or( 1<<2 );
		
		while( true )
		{
			std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
			int i;
			for( i = 1; i < this->threadsInfo.size(); ++i )
			{
				if( this->threadsInfo[i]->flags.load() & (1<<3) == 0 )
					break;
			}
			if( i == this->threadsInfo.size() )
				break;
		}
		
		for( int i = 1; i < this->threadsInfo.size(); ++i )
			delete this->threadsInfo[i];
		this->threadsInfo.clear();
		this->threads.clear();
	}
	
	threadsCount = 0;
	TrainingStrategy::Destroy();
	this->ann.Destroy();
	for( int i = 0; i < this->deltaWeights.size(); ++i )
	{
		if( this->deltaWeights[i] );
			Free( this->deltaWeights[i] );
	}
	this->deltaWeights.clear();
	
	for( int i = 0; i < this->gradient.size(); ++i )
	{
		if( this->gradient[i] );
			Free( this->gradient[i] );
	}
	this->gradient.clear();
	
	Free( this->prevDeltaWeights );
	this->learningFactor = 0.01f;
	this->prevDeltaWeights = nullptr;
	this->MSE = 0.0f;
}

BackPropagation::BackPropagation()
{
	threadsCount = 1;
	learningFactorModifier = 0.997f;
	modifyLearningFactorMaximallyOncePerEpochs = 100;
	minWeight = -1500.0f;
	maxWeight = 1500.0f;
	learningFactor = 1.0f;
	prevDeltaWeights = nullptr;
	MSE = 0.0f;
}

BackPropagation::~BackPropagation()
{
	Destroy();
}

#endif

