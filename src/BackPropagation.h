
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef BACK_PROPAGATION_H
#define BACK_PROPAGATION_H

#include "TrainingStrategy.h"

class BackPropagation : public TrainingStrategy
{
public://private:
	
	NeuralNetwork ann;
	
	float learningFactor;
	
	float * gradient;			// one per neuron
	float * deltaWeights;		// one per weight and bias
	float * prevDeltaWeights;
	
	float MSE;
	float prevMSE;
	
	float minWeight, maxWeight;
	
	sizetype modifyLearningFactorMaximallyOncePerEpochs;
	float learningFactorModifier;
	
	bool needToClampWeights;
	
	virtual unsigned LoadTrainingData( std::istream & stream ) override;
	virtual unsigned SaveTrainingData( std::ostream & stream ) const override;
	
public:
	
	inline NeuralNetwork & AccessMainNetwork();
	
	unsigned CreateSquareErrorAnalysisPerAllDataSet( const char * fileName, const Data & data, float writeOnlyGraterThan = 0.0f );
	
	inline void SetLearningModifier( float value );
	inline void SetAvailbleToChangeLearningFactorMaximallyOncePerEpochs( sizetype epochs );
	
	void SetLearningFactor( float value );
	
	inline void SetMinMaxWeights( float min, float max );
	
	virtual bool IsValid() const override;
	
	virtual void TrainOneEpoch( const Data & data ) override;
	
	void AllocateArrays();
	
	virtual float GetCurrentError() const override;
	float GetMSE() const;
	
	inline float * AccessGradient( sizetype layer, sizetype neuron );		// assume: layer > 0
	inline float * AccessDeltaWeights( sizetype layer, sizetype neuron, sizetype id );
	
	inline void CalculateOutputNeuronGradient( sizetype layer, sizetype neuron, float * desiredOutput );
	inline void CalculateHiddenNeuronGradient( sizetype layer, sizetype neuron );
	inline void UpdateWeight( sizetype layer, sizetype neuron );
	inline void UpdateDeltaWeight( sizetype layer, sizetype neuron );
	
	void ClearDeltaWeights();
	void UpdateDeltaWeights();
	void CalculateGradient( float * desiredOutput );
	void UpdateWeights( sizetype numberOfTrainedDataSets );
	
	void Init( sizetype layers, const sizetype * const neurons );
	
	virtual void Destroy() override;
	
	BackPropagation();
	~BackPropagation();
};

#include "BackPropagation.inl"

#endif

