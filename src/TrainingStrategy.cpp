
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef TRAINING_STRATEGY_CPP
#define TRAINING_STRATEGY_CPP

#include "TrainingStrategy.h"
#include "BackPropagation.h"

#include <fstream>

void TrainingStrategy::Train( const Data & data, sizetype maxEpochs, float desiredError, sizetype reportBetweenEpochs, ReportEventFunction reportEvent )
{
	sizetype i;
	for( i = 0; i < maxEpochs; ++i )
	{
		++this->currentEpoch;
		this->TrainOneEpoch( data );
		
		if( this->GetCurrentError() < desiredError )
		{
			if( reportEvent )
				reportEvent( this, i, maxEpochs, this->GetCurrentError() );
			else
				printf( "\n Epoch: %10llu max epochs: %10llu current error: %6.6f desired error: %6.6f reached ", this->currentEpoch, maxEpochs, this->GetCurrentError(), desiredError );
			return;
		}
		
		if( reportBetweenEpochs )
		{
			if( i % reportBetweenEpochs == reportBetweenEpochs-1 )
			{
				if( reportEvent )
					reportEvent( this, i, maxEpochs, this->GetCurrentError() );
				else
					printf( "\n Epoch: %10llu current error: %6.6f learning factor = %6.6f ", this->currentEpoch, this->GetCurrentError(), ((BackPropagation*)this)->learningFactor );
			}
		}
	}
}

unsigned TrainingStrategy::LoadTrainingData( std::istream & stream )
{
	stream >> this->currentEpoch;
	return 0;
}

unsigned TrainingStrategy::SaveTrainingData( std::ostream & stream ) const
{
	stream << this->currentEpoch << "\n";
	return 0;
}

unsigned TrainingStrategy::LoadTrainingDataFromFile( const char * fileName )
{
	if( fileName )
	{
		std::ifstream file( fileName );
		if( file.good() )
			return this->LoadTrainingData( file );
		else
			return 2;
	}
	
	return 1;
}

unsigned TrainingStrategy::SaveTrainingDataToFile( const char * fileName ) const
{
	if( fileName )
	{
		std::ofstream file( fileName );
		if( file.good() )
			return this->SaveTrainingData( file );
		else
			return 2;
	}
	
	return 1;
}

void TrainingStrategy::Destroy()
{
	this->currentEpoch = 0;
}

TrainingStrategy::TrainingStrategy()
{
	currentEpoch = 0;
}

TrainingStrategy::~TrainingStrategy()
{
	Destroy();
}

#endif

