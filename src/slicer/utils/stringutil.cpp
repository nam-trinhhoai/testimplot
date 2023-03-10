/*
 * StringUtil.cpp
 *
 *  Created on: Apr 5, 2020
 *      Author: l0222891
 */

#include "stringutil.h"

#include <algorithm>

bool endsWith(const std::string & s, const std::string & suffix) {
     return s.rfind(suffix) == s.length() - suffix.length();
}

// trim from start (in place)
void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                return !std::isspace(ch);
        }));
}

// trim from end (in place)
void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                return !std::isspace(ch);
        }).base(), s.end());
}

// trim from both ends (in place)
void trim(std::string &s)
{
        ltrim(s);
        rtrim(s);
}

// trim from start (copying)
std::string ltrim_copy(std::string s) {
        ltrim(s);
        return s;
}

// trim from end (copying)
std::string rtrim_copy(std::string s) {
        rtrim(s);
        return s;
}

// trim from both ends (copying)
std::string trim_copy(std::string s) {
        trim(s);
        return s;
}


