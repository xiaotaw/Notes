/*
 The test code on windows(unfinished).

 There are still some bug in this code.

 And compile tensorflow on windows is a very tricky wrok.

 More time will be wasted on it.

*/

#ifdef _WIN32
#define COMPILER_MSVC
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <string>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

class HardwareConfig {
public:
	int m_parallelism_threads;
	bool m_allow_growth;
	std::string m_visible_device_list;
	float m_per_process_gpu_memory_fraction;

	HardwareConfig(){
		m_parallelism_threads = 0;
		m_allow_growth = false;
		m_visible_device_list = "0";
		m_per_process_gpu_memory_fraction = 1.0f;
	}

	HardwareConfig(int pt, bool b, std::string s, float f):
		m_parallelism_threads(pt), 
		m_allow_growth(b),
		m_visible_device_list(s),
		m_per_process_gpu_memory_fraction(f){
		;
	}
};

bool LoadGraphFromFile(const std::string & file_name, tensorflow::GraphDef graphdef) {
	tensorflow::Status status = ReadBinaryProto(tensorflow::Env::Default(), file_name, &graphdef);
	if (!status.ok()) {
		std::cout << "load graph failed! " << status.ToString() << std::endl;
		return false;
	}
	return true;
}

bool FileExist(const std::string file_name) {
	std::ifstream in_f;
	in_f.open(file_name, std::ios::binary);
	if (in_f) {
		in_f.close();
		return true;
	}
	return false;

}

bool CreateTFSession(tensorflow::Session ** session, const tensorflow::GraphDef & graphdef, HardwareConfig hardware_config) {

	tensorflow::SessionOptions options = tensorflow::SessionOptions();

	// options.config.gpu_options() 在windows下会有_GPUOptions_default_instance符号不可见的问题
	// 在linux下没有问题，但是visible_device_list()无内容
	//std::cout << "initial visible_device_list: " << options.config.gpu_options().visible_device_list() << std::endl;
	options.config.mutable_gpu_options()->set_visible_device_list(hardware_config.m_visible_device_list);

	options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(hardware_config.m_per_process_gpu_memory_fraction);
	
	tensorflow::Status status = NewSession(options, (tensorflow::Session**)session);
	if (!status.ok()) {
		std::cout << "Error: failed to create session!" << std::endl;
		return false;
	}
	status = (*session)->Create(graphdef);
	if (!status.ok()) {
		std::cout << "Error: failed to create graph!" << std::endl;
		return false;
	}

	return true;

}

int main() {
	HardwareConfig session_option(0, true, "0", 0.1);
	tensorflow::GraphDef graphdef;
	tensorflow::Session * session = NULL;
	tensorflow::Status status;

	const std::string file_name = "E:\\xiaotao\\test_c\\vs_project\\Test_GPUOption\\graph.pb";
	//const std::string file_name = "graph.pb";
	if (! FileExist(file_name)) {
		std::cout << "file not exists!" << std::endl;
		//return -1;
	}
	status = ReadBinaryProto(tensorflow::Env::Default(), file_name, &graphdef);
	if (!status.ok()) {
		std::cout << "load graph failed! " << status.ToString() << std::endl;
		//return false;
	}
	//LoadGraphFromFile("E:\\xiaotao\\test_c\\vs_project\\Test_GPUOption\\graph.pb", graphdef);

	CreateTFSession(&session, graphdef, session_option);

	/*
	// Setup inputs and outputs:

	// Our graph doesn't require any inputs, since it specifies default values,
	// but we'll change an input to demonstrate.
	tensorflow::Tensor a(tensorflow::DT_FLOAT, tensorflow::TensorShape());
	a.scalar<float>()() = 3.0;

	tensorflow::Tensor b(tensorflow::DT_FLOAT, tensorflow::TensorShape());
	b.scalar<float>()() = 2.0;

	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ "a", a },
		{ "b", b },
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "c" }, {}, &outputs);
	std::cout << "sess run (compute c) ... " << status.ToString() << "\n";
	if (!status.ok()) {
		return 1;
	}

	// Grab the first output (we only evaluated one graph node: "c")
	// and convert the node to a scalar representation.
	auto output_c = outputs[0].scalar<float>();

	// (There are similar methods for vectors and matrices here:
	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

	// Print the results
	std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
	std::cout << output_c() << "\n"; // 30

	*/
	// Free any resources used by the session
	session->Close();

	return 0;
}
