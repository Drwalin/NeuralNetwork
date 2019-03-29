
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef TRAINING_STRATEGY_H
#define TRAINING_STRATEGY_H

#include "NeuralNetwork.h"
#include "Data.h"
#include "Memory.h"

#include <vector>
#include <iostream>

typedef void (*ReportEventFunction)( class TrainingStrategy*, const Data * data, sizetype epoch, sizetype maxEpochs, float currentMSE );

class TrainingStrategy
{
protected:
	
	sizetype currentEpoch;
	
	virtual unsigned LoadTrainingData( std::istream & stream );
	virtual unsigned SaveTrainingData( std::ostream & stream ) const;
	
public:
	
	sizetype GetCurrentEpoch() const;
	
	virtual bool IsValid() const = 0;
	
	virtual float GetCurrentError() const = 0;
	
	unsigned LoadTrainingDataFromFile( const char * fileName );
	unsigned SaveTrainingDataToFile( const char * fileName ) const;
	
	virtual void TrainOneEpoch( const Data & data ) = 0;
	
	void Train( const Data & data, sizetype maxEpochs, float desiredError, sizetype reportBetweenEpochs = 0, ReportEventFunction reportEvent = nullptr );		// reportBetweenEpochs = 0 - no report
	
	virtual void Destroy();
	
	TrainingStrategy();
	~TrainingStrategy();
};

#include "TrainingStrategy.inl"

#endif

