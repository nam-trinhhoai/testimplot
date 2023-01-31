#ifndef IOUTIL_H
#define IOUTIL_H

#include <cstddef>
#include <cstdint>
#include <string>

template<size_t>
inline void __switch_list_endianness_inplace(void* src, const size_t nmemb);

void switch_list_endianness_inplace(void* data, const size_t smemb, const size_t nmemb);

std::string remove_extension(const std::string& path);

bool isFileReadable(const char * file);

/**
 * brick2DToContinuousData convert a 2D bricked data to a continuous data
 * param: inputFirst and inputEnd represent an 1D array of size sizeX*sizeY containing a list of bricks
 * param: outputFirst and outputEnd represent an 1D array of size sizeX*sizeY
 * param: brickSizeX brick size along quickest axis
 * param: brickSizeY brick size along slowest axis
 * param: sizeX size along the quickest axis
 * param: sizeY size along the slowest axis
 */
template<typename InputIterator, typename OutputIterator>
bool brick2DToContinuousData(InputIterator inputFirst, InputIterator inputEnd, OutputIterator outputFirst,
		OutputIterator outputEnd, long brickSizeX, long brickSizeY, long sizeX, long sizeY);

#include "ioutil.hpp"

#endif
