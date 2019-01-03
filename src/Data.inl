
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef DATA_INL
#define DATA_INL

inline DataSet & Data::operator [] ( sizetype id )
{
	return this->data[id];
}

inline const DataSet & Data::operator [] ( sizetype id ) const
{
	return this->data[id];
}

inline sizetype Data::Size() const
{
	return this->data.size();
}

inline sizetype Data::GetInputs() const
{
	return this->inputs;
}

inline sizetype Data::GetOutputs() const
{
	return this->outputs;
}

inline sizetype Data::DataSetsNumber() const
{
	return this->data.size();
}

#endif

