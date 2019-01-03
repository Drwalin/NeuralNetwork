
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef DATASET_CPP
#define DATASET_CPP

#include "DataSet.h"

#include <fstream>

#include <cassert>

unsigned DataSet::LoadFromFile( const char * fileName )
{
	if( fileName )
	{
		std::ifstream file( fileName );
		if( file.good() )
		{
			sizetype ret = this->LoadFromStandardStream( file );
			printf( "\n Load From File = %u ", ret );
			file.close();
			return ret;
		}
	}
	return 1;
}

unsigned DataSet::SaveToFile( const char * fileName ) const
{
	if( fileName )
	{
		std::ofstream file( fileName );
		if( file.good() )
		{
			sizetype ret = this->SaveToStandardStream( file );
			printf( "\n Load From File = %u ", ret );
			file.close();
			return ret;
		}
	}
	return 1;
}

unsigned DataSet::LoadFromStandardStream( std::istream & stream )
{
	if( !stream.good() )
		return 2;
	
	stream >> this->inputs;
	stream >> this->outputs;
	
	if( stream.eof() || stream.fail() ) { this->Destroy(); return 3; }
	
	this->input = Alloc<float>( this->inputs );
	this->output = Alloc<float>( this->outputs );
	assert( this->input );
	assert( this->output );
	
	sizetype i;
	
	for( i = 0; i < this->inputs; ++i )
		stream >> this->input[i];
	if( stream.eof() || stream.fail() ) { this->Destroy(); return 4; }
	
	for( i = 0; i < this->outputs; ++i )
		stream >> this->output[i];
	if( stream.fail() ) { this->Destroy(); return 5; }
	
	return 0;
}

unsigned DataSet::SaveToStandardStream( std::ostream & stream ) const
{
	if( !this->IsValid() )
	{
		printf( "\n Saving invalid data set: { %llu, %llu, %llu, %llu } ", this->inputs, this->outputs, this->input, this->output );
		return 2;
	}
	
	if( !stream.good() )
		return 3;
	
	stream << this->inputs << "\n";
	stream << this->outputs << "\n";
	
	if( stream.fail() ) { return 4; }
	
	sizetype i;
	
	for( i = 0; i < this->inputs; ++i )
		stream << this->input[i] << " ";
	stream << "\n";
	if( stream.fail() ) { return 5; }
	
	for( i = 0; i < this->outputs; ++i )
		stream << this->output[i] << " ";
	if( stream.fail() ) { return 6; }
	
	return 0;
}


unsigned DataSet::LoadFromStandardStreamNoNumberOfInputsData( std::istream & stream, sizetype in, sizetype out )
{
	if( !stream.good() )
		return 2;
	
	this->inputs = in;
	this->outputs = out;
	
	this->input = Alloc<float>( this->inputs );
	this->output = Alloc<float>( this->outputs );
	assert( this->input );
	assert( this->output );
	
	sizetype i;
	
	for( i = 0; i < this->inputs; ++i )
		stream >> this->input[i];
	if( stream.eof() || stream.fail() ) { this->Destroy(); return 4; }
	
	for( i = 0; i < this->outputs; ++i )
		stream >> this->output[i];
	if( stream.fail() ) { this->Destroy(); return 5; }
	
	return 0;
}

unsigned DataSet::SaveToStandardStreamNoNumberOfInputsData( std::ostream & stream ) const
{
	if( !this->IsValid() )
	{
		printf( "\n Saving invalid data set: { %llu, %llu, %llu, %llu } ", this->inputs, this->outputs, this->input, this->output );
		return 2;
	}
	
	if( !stream.good() )
		return 3;
	
	sizetype i;
	
	for( i = 0; i < this->inputs; ++i )
		stream << this->input[i] << " ";
	stream << "\n";
	if( stream.fail() ) { return 5; }
	
	for( i = 0; i < this->outputs; ++i )
		stream << this->output[i] << " ";
	if( stream.fail() ) { return 6; }
	
	return 0;
}

void DataSet::Destroy()
{
	Free( input );
	Free( output );
	
	input = nullptr;
	output = nullptr;
	
	inputs = 0;
	outputs = 0;
}

unsigned DataSet::operator = ( const DataSet & other )
{
	this->Destroy();
	
	if( other.IsValid() )
	{
		this->inputs = other.inputs;
		this->outputs = other.outputs;
		
		this->input = Alloc<float>( this->inputs );
		this->output = Alloc<float>( this->outputs );
		assert( this->input );
		assert( this->output );
		
		sizetype i;
		for( i = 0; i < this->inputs; ++i )
			this->input[i] = other.input[i];
		
		for( i = 0; i < this->outputs; ++i )
			this->output[i] = other.output[i];
		
		return 0;
	}
	else
	{
		printf( "\n Error: DataSet::operator =" );
	}
	
	return 1;
}

DataSet::DataSet( sizetype in, sizetype out, const float * ins, const float * outs )
{
	input = nullptr;
	output = nullptr;
	
	inputs = 0;
	outputs = 0;
	
	if( ins != nullptr && outs != nullptr && in != 0 && out != 0 )
	{
		this->inputs = in;
		this->outputs = out;
		
		this->input = Alloc<float>( this->inputs );
		this->output = Alloc<float>( this->outputs );
		assert( this->input );
		assert( this->output );
		
		sizetype i;
		for( i = 0; i < this->inputs; ++i )
			this->input[i] = ins[i];
		
		for( i = 0; i < this->outputs; ++i )
			this->output[i] = outs[i];
	}
}

DataSet::DataSet( const DataSet * other )
{
	input = nullptr;
	output = nullptr;
	
	inputs = 0;
	outputs = 0;
	
	if( other && other->IsValid() )
	{
		this->inputs = other->inputs;
		this->outputs = other->outputs;
		
		this->input = Alloc<float>( this->inputs );
		this->output = Alloc<float>( this->outputs );
		assert( this->input );
		assert( this->output );
	
		sizetype i;
		for( i = 0; i < this->inputs; ++i )
			this->input[i] = other->input[i];
		
		for( i = 0; i < this->outputs; ++i )
			this->output[i] = other->output[i];
	}
}

DataSet::DataSet( const DataSet & other )
{
	input = nullptr;
	output = nullptr;
	
	inputs = 0;
	outputs = 0;
	
	if( other.IsValid() )
	{
		this->inputs = other.inputs;
		this->outputs = other.outputs;
		
		this->input = Alloc<float>( this->inputs );
		this->output = Alloc<float>( this->outputs );
		assert( this->input );
		assert( this->output );
		
			sizetype i;
			for( i = 0; i < this->inputs; ++i )
				this->input[i] = other.input[i];
			
			for( i = 0; i < this->outputs; ++i )
				this->output[i] = other.output[i];
	}
}

DataSet::DataSet()
{
	input = nullptr;
	output = nullptr;
	
	inputs = 0;
	outputs = 0;
}

DataSet::~DataSet()
{
	Destroy();
}

#endif

