
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef BACK_PROPAGATION_H
#define BACK_PROPAGATION_H

#include <vector>
#include <thread>
#include <atomic>
#include <random>

#include "TrainingStrategy.h"


class BackPropagation : public TrainingStrategy
{
public://private:
	
	class BackPropagationThreadInfo
	{
	public:
		std::atomic<void*> data;
		float MSE;
		sizetype threadsCount;
		sizetype threadID;
		BackPropagation * backPropagation;
		std::atomic<unsigned> flags;
		/*
			flags:
				[0] - queue compute
				[1] - ended computing
				
				[2] - queue end
				[3] - ended
				
				[4] - weights updated (used only in batch mode)
		*/
		
		std::default_random_engine generator;
		std::uniform_int_distribution<sizetype> distribution;
		
		sizetype Random();
		BackPropagationThreadInfo( sizetype threadID );
		void InitDistribution( sizetype dataSetSize );
	};
	
	std::atomic<sizetype> currentDataSetCount;
	std::atomic<sizetype> currentBatchCount;
	sizetype batchSize;
	
	NeuralNetwork ann;
	
	float learningFactor;
	
	std::vector < float* > gradient;			// one per neuron
	std::vector < float* > deltaWeights;		// one per weight and bias
	float * prevDeltaWeights;
	
	float MSE;
	float prevMSE;
	
	float minWeight, maxWeight;
	
	sizetype modifyLearningFactorMaximallyOncePerEpochs;
	float learningFactorModifier;
	
	bool needToClampWeights;
	
	virtual unsigned LoadTrainingData( std::istream & stream ) override;
	virtual unsigned SaveTrainingData( std::ostream & stream ) const override;
	
	sizetype threadsCount;
	std::vector < std::thread* > threads;
	std::vector < BackPropagationThreadInfo* > threadsInfo;
	
	const DataSet * GetNextRandomDataSet( const Data * data, sizetype threadID );
	
public:
	
	void ThreadFunction( sizetype threadID );
	void ThreadFunctionBatch( sizetype threadID );
	void ThreadFunctionWeightsUpdateBatch( sizetype threadID );
	
	float GetLearningFactor() const;
	
	inline NeuralNetwork & AccessMainNetwork();
	
	unsigned CreateSquareErrorAnalysisPerAllDataSet( const char * fileName, const Data & data, float writeOnlyGraterThan = 0.0f );
	
	inline void SetLearningModifier( float value );
	inline void SetAvailbleToChangeLearningFactorMaximallyOncePerEpochs( sizetype epochs );
	
	void SetLearningFactor( float value );
	
	inline void SetMinMaxWeights( float min, float max );
	
	virtual bool IsValid() const override;
	
	virtual void TrainOneEpoch( const Data & data ) override;
	
	void AllocateArrays( sizetype arraysCount );
	
	virtual float GetCurrentError() const override;
	float GetMSE() const;
	
	inline float * AccessGradient( sizetype layer, sizetype neuron, float * gradient );		// assume: layer > 0
	inline float * AccessDeltaWeights( sizetype layer, sizetype neuron, sizetype id, float * deltaWeights );
	
	inline void CalculateOutputNeuronGradient( sizetype layer, sizetype neuron, float * desiredOutput, float * gradient, sizetype threadID );
	inline void CalculateHiddenNeuronGradient( sizetype layer, sizetype neuron, float * gradient_, sizetype threadID );
	inline void UpdateWeight( sizetype layer, sizetype neuron );
	inline void UpdateDeltaWeight( sizetype layer, sizetype neuron, float * gradient_, float * deltaWeights_, sizetype threadID );
	
	void ClearDeltaWeights( sizetype threadID );
	void UpdateDeltaWeights( float * gradient_, float * deltaWeights_, sizetype threadID );
	void CalculateGradient( float * desiredOutput, sizetype threadID );
	void UpdateWeights( sizetype numberOfTrainedDataSets, sizetype threadsCount, sizetype threadID );
	
	void SetBatchSize( sizetype batch );
	void Init( sizetype layers, const sizetype * const neurons );
	
	void PreInit( sizetype threadsCount = 1 );
	
	virtual void Destroy() override;
	
	BackPropagation();
	~BackPropagation();
};

void BackPropagationThreadFunction( BackPropagation::BackPropagationThreadInfo * threadInfo );

#include "BackPropagation.inl"

#endif

