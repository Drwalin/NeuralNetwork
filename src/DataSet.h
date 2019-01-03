
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef DATASET_H
#define DATASET_H

#include <iostream>

#include "Memory.h"

class DataSet
{
private:
	
	sizetype inputs;
	sizetype outputs;
	
	float * input;
	float * output;
	
	friend class TrainingStrategy;
	
public:
	
	inline sizetype GetInputs() const;
	inline sizetype GetOutputs() const;
	
	inline float * GetInputPointer() const;
	inline float * GetOutputPointer() const;
	
	inline bool IsValid() const;
	
	unsigned LoadFromFile( const char * fileName );
	unsigned SaveToFile( const char * fileName ) const;
	unsigned LoadFromStandardStream( std::istream & stream );
	unsigned SaveToStandardStream( std::ostream & stream ) const;
	unsigned LoadFromStandardStreamNoNumberOfInputsData( std::istream & stream, sizetype in, sizetype out );
	unsigned SaveToStandardStreamNoNumberOfInputsData( std::ostream & stream ) const;
	
	void Destroy();
	
	unsigned operator = ( const DataSet & other );
	
	DataSet( sizetype in, sizetype out, const float * ins, const float * outs );
	DataSet( const DataSet * other );
	DataSet( const DataSet & other );
	DataSet();
	~DataSet();
};

#include "DataSet.inl"

#endif

