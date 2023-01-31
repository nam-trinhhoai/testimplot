#ifndef SAMPLE_TYPE_BINDER_H
#define SAMPLE_TYPE_BINDER_H

#include "imageformats.h"

#include <functional>

class SampleTypeBinder{
    ImageFormats::QSampleType type;
public:
    SampleTypeBinder(ImageFormats::QSampleType type){
        this->type = type;
    }
    SampleTypeBinder(const SampleTypeBinder& binder){
        this->type = binder.type;
    }
    ~SampleTypeBinder(){}

    /** 
	 * @brief  bind and call the given functor with the good template type
	 * @note   the functor is expected to have a function that match with the following signature
	 * 
	 * template<typename InputType>
	 *	struct RunFilter {
	 *		static ResultType run(Args ... ){
	 *			return ResultType();
	 *		}
	 *	};
	 * 
	 * IMPORTANT : this is only valid when the return type of the functor is always the same even for different input types
	 * 
	 * Throws invalid_argument exception if the binding could not be performed
	 */
	template< template<typename...> typename TemplatedFunctor, typename... Args>
	auto bind(Args&&... args)->decltype(TemplatedFunctor<void>::run(std::forward<Args>(args)...)){
		return bind<TemplatedFunctor>(type,std::forward<Args>(args)...);
	}


	 template <template<typename,unsigned int> class TemplatedFunctor,unsigned int Dims,class... Args>
	 std::function<bool()> bindFunctor( Args&&... args)
	{
		 switch(type){
				case ImageFormats::QSampleType::UINT8:
					return std::bind(&TemplatedFunctor<unsigned char, Dims>::run,args...);
				case ImageFormats::QSampleType::INT8:
					return std::bind(&TemplatedFunctor<char, Dims>::run,args...);
				case ImageFormats::QSampleType::UINT16:
					return std::bind(&TemplatedFunctor<unsigned short, Dims>::run,args...);
				case ImageFormats::QSampleType::INT16:
					return std::bind(&TemplatedFunctor<short, Dims>::run,args...);
				case ImageFormats::QSampleType::INT32:
					return std::bind(&TemplatedFunctor<int, Dims>::run,args...);
				case ImageFormats::QSampleType::UINT32:
					return std::bind(&TemplatedFunctor<unsigned int, Dims>::run,args...);
				case ImageFormats::QSampleType::FLOAT32:
					return std::bind(&TemplatedFunctor<float, Dims>::run,args...);
				case ImageFormats::QSampleType::FLOAT64:
					return std::bind(&TemplatedFunctor<double, Dims>::run,args...);
			}
			throw std::invalid_argument("The given sample type cannot be used");
	}

private:
	//for the decltype we consider that 
	template<template<typename...> typename TemplatedFunctor, typename... Args>
	static auto bind(const ImageFormats::QSampleType type, Args&&... args) -> decltype(TemplatedFunctor<void>::run(std::forward<Args>(args)...)){
		switch(type){
			case ImageFormats::QSampleType::UINT8:
				return TemplatedFunctor<unsigned char>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::INT8:
				return TemplatedFunctor<signed char>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::UINT16:
				return TemplatedFunctor<unsigned short>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::INT16:
				return TemplatedFunctor<short>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::INT32:
				return TemplatedFunctor<int>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::UINT32:
				return TemplatedFunctor<unsigned int>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::FLOAT32:
				return TemplatedFunctor<float>::run(std::forward<Args>(args)...);
			case ImageFormats::QSampleType::FLOAT64:
				return TemplatedFunctor<double>::run(std::forward<Args>(args)...);
		}
		throw std::invalid_argument("The given sample type cannot be used");
	}
	
};

#endif /* SAMPLE_TYPE_BINDER__H */
