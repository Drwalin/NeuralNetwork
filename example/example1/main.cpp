
#include "../../src/NeuralNetwork.cpp"
#include "../../src/Random.cpp"

#include <cstdlib>
#include <ctime>

int main()
{
	srand( time( nullptr ) );
	
	sizetype i = 0, j, k;
	
	for( i = 0; i < 10; ++i )
	{
		sizetype layers = ( rand() % 7 ) + 3;
		sizetype neurons[ layers ];
		for( j = 0; j < layers; ++j )
		{
			neurons[j] = (rand()%30) + 3;
		}
		
		float input[neurons[0]];
		float output[neurons[layers-1]];
		
		for( j = 0; j < neurons[0]; ++j )
		{
			input[j] = Random::Random( 30.0, 10.0 );
		}
		
		{
			NeuralNetwork net;
			
			net.Init( layers, neurons );
			
			net.InitRandom();
			
			net.Run( &input[0] );
			
			net.GetOutputs();
			
			memcpy( output, net.GetOutputs(), neurons[layers-1] * sizeof(float) );
			
			net.SaveToFile( "temp.file" );
			
			net.Destroy();
		}
		
		{
			NeuralNetwork net;
			
			net.LoadFromFile( "temp.file" );
			
			net.Run( &input[0] );
			
			printf( "\n SE between both: %f ", net.GetSE( output ) );
			
			net.Destroy();
		}
	}
	
	return 0;
}



