
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef DATA_CPP
#define DATA_CPP

#include "Data.h"

#include <fstream>

void Data::JoinData( const Data & data )
{
	if( this->inputs == data.inputs && this->outputs == data.outputs )
	{
		sizetype prevsize = this->data.size();
		this->data.resize( prevsize + data.data.size() );
		
		sizetype i;
		for( i = prevsize; i < this->data.size(); ++i )
			this->data[i] = data.data[i-prevsize];
	}
	this->datasize = this->data.size();
}

void Data::AddDataSet( const DataSet & dataSet )
{
	if( ( ( dataSet.GetInputs() == this->inputs && dataSet.GetOutputs() == this->outputs ) || this->data.size() == 0 ) && dataSet.IsValid() )
	{
		this->inputs = dataSet.GetInputs();
		this->outputs = dataSet.GetOutputs();
		this->data.resize( this->data.size() + 1 );
		this->data.back() = dataSet;
	}
	else
	{
		printf( "\n Error in Data::AddDataSet" );
	}
	this->datasize = this->data.size();
}

unsigned Data::AppendLoadDataSetsFromFile( const char * fileName )
{
	sizetype i;
	
	if( fileName )
	{
		std::ifstream file( fileName );
		if( file.good() )
		{
			sizetype tests;
			sizetype inputs, outputs;
			
			file >> inputs;
			file >> outputs;
			
			if( inputs != this->inputs && this->inputs != 0 )
			{
				this->datasize = this->data.size();
				return 4;
			}
			this->inputs = inputs;
			
			if( outputs != this->outputs && this->outputs != 0 )
			{
				this->datasize = this->data.size();
				return 5;
			}
			this->outputs = outputs;
			
			file >> tests;
			
			sizetype prevSize = this->data.size();
			tests += this->data.size();
			
			this->data.resize( tests );
			
			for( i = prevSize; i < tests; ++i )
			{
				if( this->data[i].LoadFromStandardStreamNoNumberOfInputsData( file, this->inputs, this->outputs ) != 0 )
				{
					file.close();
					this->data.resize( i );
					this->data.shrink_to_fit();
					this->datasize = this->data.size();
					return 3;
				}
				else if( this->data[i].GetInputs() != this->inputs || this->data[i].GetOutputs() != this->outputs )
				{
					--i;
					--tests;
				}
			}
			
			this->data.resize( tests );
			this->data.shrink_to_fit();
			
			file.close();
			
			this->datasize = this->data.size();
			return 0;
		}
		else
		{
			this->datasize = this->data.size();
			return 2;
		}
	}
	
	this->datasize = this->data.size();
	return 1;
}

unsigned Data::LoadDataSetsFromFile( const char * fileName )
{
	this->Destroy();
	this->datasize = this->data.size();
	return this->AppendLoadDataSetsFromFile( fileName );
}

unsigned Data::SaveDataSetsToFile( const char * fileName )
{
	if( fileName )
	{
		std::ofstream file( fileName );
		if( file.good() )
		{
			sizetype i, err;
			
			file << this->inputs << "\n";
			file << this->outputs << "\n";
			
			file << this->data.size() << "\n";
			
			for( i = 0; i < this->data.size(); ++i )
			{
				err = this->data[i].SaveToStandardStreamNoNumberOfInputsData( file );
				if( err != 0 )
				{
					printf( "\n Saving data sets to file stopped in iteration: %llu   with error: %llu ", i, err );
					file.close();
					return 3;
				}
				file << "\n";
			}
			
			file.close();
			return 0;
		}
		else
		{
			return 2;
		}
	}
	
	return 1;
}

void Data::RemoveInvalidDataSets()
{
	sizetype i;
	for( i = 0; i < this->data.size(); ++i )
	{
		if( this->data[i].GetInputs() != this->inputs || this->data[i].GetOutputs() != this->outputs )
		{
			this->data.erase( this->data.begin() + i );
			--i;
		}
	}
	
	this->data.shrink_to_fit();
	this->datasize = this->data.size();
}

void Data::SetInputsAndOutputs( sizetype inputs, sizetype outputs )
{
	this->inputs = inputs;
	this->outputs = outputs;
	this->RemoveInvalidDataSets();
	this->datasize = this->data.size();
}

void Data::Destroy()
{
	sizetype i;
	for( i = 0; i < this->data.size(); ++i )
		this->data[i].Destroy();
	this->data.clear();
	this->inputs = 0;
	this->outputs = 0;
	this->datasize = this->data.size();
}

Data::Data()
{
	inputs = 0;
	outputs = 0;
	this->datasize = this->data.size();
}

Data::~Data()
{
	Destroy();
	this->datasize = this->data.size();
}

#endif

