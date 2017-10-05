#include <iostream>
#include <string>

namespace pfc {
	namespace cuda {

		class CudaError : public std::runtime_error {
		public:
			CudaError(std::string msg, std::string file, std::string lineNum): std::runtime_error(getErrorMsg(msg, file, lineNum))

			static std::string getErrorMsg(std::string msg, std::string file, std::string lineNum) {
				return msg + "" + " " + file + " " + lineNum;
			}
		};
		void check(cudaError_t &err, std::string msg, std::string file, int lineNum) {
			if (err != cudaSuccess) {
				throw CudaError(msg, file, lineNum);
			}
		}
	}
}