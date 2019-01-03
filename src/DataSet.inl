
/*
	Copyright (C) 2019 Marek Zalewski aka Drwalin - All Rights Reserved
	
	Any one is authorized to copy, use or modify this file,
	but only when this file is marked as written by Marek Zalewski.
	
	No one can claim this file is originally written by them.
	
	No one can modify or remove this Copyright notice from this file.
*/

#ifndef DATASET_INL
#define DATASET_INL

inline bool DataSet::IsValid() const
{
	return this->input && this->output && this->inputs && this->outputs;
}

inline sizetype DataSet::GetInputs() const
{
	return this->inputs;
}

inline sizetype DataSet::GetOutputs() const
{
	return this->outputs;
}

inline float * DataSet::GetInputPointer() const
{
	return this->input;
}

inline float * DataSet::GetOutputPointer() const
{
	return this->output;
}

#endif

