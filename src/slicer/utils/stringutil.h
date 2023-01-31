/*
 * StringUtil.h
 *
 *  Created on: Apr 5, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_UTILS_STRINGUTIL_H_
#define TARUMAPP_SRC_UTILS_STRINGUTIL_H_

#include <iostream>
#include <string>

bool endsWith(const std::string & s, const std::string & suffix);

// trim from start (in place)
void ltrim(std::string &s);

// trim from end (in place)
void rtrim(std::string &s);

// trim from both ends (in place)
void trim(std::string &s);

// trim from start (copying)
 std::string ltrim_copy(std::string s);

// trim from end (copying)
std::string rtrim_copy(std::string s);

// trim from both ends (copying)
 std::string trim_copy(std::string s);


#endif /* TARUMAPP_SRC_UTILS_STRINGUTIL_H_ */
