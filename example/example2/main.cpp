
#include "../../src/BackPropagation.cpp"
#include "../../src/TrainingStrategy.cpp"
#include "../../src/NeuralNetwork.cpp"
#include "../../src/DataSet.cpp"
#include "../../src/Random.cpp"
#include "../../src/Data.cpp"

#include <cstdio>

#include <string>

sizetype i;

void Report( class TrainingStrategy*, sizetype epoch, sizetype maxEpochs, float currentMSE )
{
	if( i % 100 == 0 )
		printf( "\n [%llu]: Epoch = %7llu   ; MSE = %f ", i, epoch, currentMSE );
}

int main()
{
	Data dataSet;
	sizetype ret;
	ret = dataSet.LoadDataSetsFromFile( "xor.data" );
	if( ret == 0 )
	{
		printf( "\n Xor data set loaded correctly" );
	}
	else
	{
		printf( "\n Xor data set couldn't be loaded correctly; ret = %llu ", ret );
		return 0;
	}
	
	BackPropagation training;
	
	sizetype neurons[] = { 2, 3, 1 };
	
	for( i = 0; i < 1000000; ++i )
	{
		training.Init( 3, neurons );
		
		training.AccessMainNetwork().InitRandom();
		
		training.SetLearningFactor( 23.0f );
		training.SetAvailbleToChangeLearningFactorMaximallyOncePerEpochs( 10 );
		
		training.Train( dataSet, 1000000, 0.001f, 1000000, Report );
		
		while( false )
		{
			float in[2];
			scanf( "%f%f", in, in+1 );
			training.AccessMainNetwork().Run( in );
			printf( "\n Output: %f ", *training.AccessMainNetwork().GetOutputs() );
			training.Train( dataSet, 1000000, 0.00001f, 900 );
		}
		
		training.Destroy();
	}
	
	return 0;
}



