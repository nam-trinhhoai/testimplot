#include "ioutil.h"

#include <boost/filesystem.hpp>

template<> inline void __switch_list_endianness_inplace<2>(void* src, const size_t nmemb){
	uint16_t* _src = (uint16_t*)src;
	for(int i=0;i<nmemb;++i){
		_src[i]=__builtin_bswap16(_src[i]);
	}
}

template<> inline void __switch_list_endianness_inplace<4>(void* src, const size_t nmemb){
	uint32_t* _src = (uint32_t*)src;
	for(int i=0;i<nmemb;++i){
		_src[i]=__builtin_bswap32(_src[i]);
	}
}

template<> inline void __switch_list_endianness_inplace<8>(void* src, const size_t nmemb){
	uint64_t* _src = (uint64_t*)src;
	for(int i=0;i<nmemb;++i){
		_src[i]=__builtin_bswap64(_src[i]);
	}
}

void switch_list_endianness_inplace(void* data, const size_t smemb, const size_t nmemb)
{
	switch(smemb){
	case 2:__switch_list_endianness_inplace<2>(data,nmemb);break;
	case 4:__switch_list_endianness_inplace<4>(data,nmemb);break;
	case 8:__switch_list_endianness_inplace<8>(data,nmemb);break;
	default: break;
	}
}

std::string remove_extension(const std::string& path) {
    if (path == "." || path == "..")
        return path;

    size_t pos = path.find_last_of("\\/.");
    if (pos != std::string::npos && path[pos] == '.')
        return path.substr(0, pos);

    return path;
}

bool isFileReadable(const char * file) {
        return access(file, R_OK) == 0;
}
