#include <poplar/Vertex.hpp>

class TraceVertex : public poplar::Vertex {
public:
    // Fields
    poplar::Input<Vector<float, POPC_">> raysIn;
    poplar::Output<Vector<float, POPC_">> raysOut;

    // Compute function
    bool compute() {
        
        return true;
    }
};