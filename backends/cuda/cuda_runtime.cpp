namespace ember {

    std::unique_ptr<IRuntime> RuntimeFactory::create_cuda() {
        return std::make_unique<cuda::CudaRuntime>();
    }
}