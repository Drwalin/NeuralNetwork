
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef DATA_H
#define DATA_H

#include "DataSet.h"
#include "Memory.h"

#include <vector>

class Data
{
private:
	
	sizetype inputs;
	sizetype outputs;
	std::vector < DataSet > data;
	sizetype datasize;
	
public:
	
	inline sizetype Size() const;
	inline DataSet & operator [] ( sizetype id );
	inline const DataSet & operator [] ( sizetype id ) const;
	
	inline sizetype GetInputs() const;
	inline sizetype GetOutputs() const;
	
	inline sizetype DataSetsNumber() const;
	
	void AddDataSet( const DataSet & dataSet );
	
	void JoinData( const Data & data );
	
	unsigned LoadDataSetsFromFile( const char * fileName );
	unsigned SaveDataSetsToFile( const char * fileName );
	unsigned AppendLoadDataSetsFromFile( const char * fileName );
	
	void RemoveInvalidDataSets();
	
	void SetInputsAndOutputs( sizetype inputs, sizetype outputs );
	
	void Destroy();
	
	Data();
	~Data();
};

#include "Data.inl"

#endif

