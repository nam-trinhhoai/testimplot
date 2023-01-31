#include "datasethashkey.h"
#include <sstream>
#include <iostream>

DatasetHashKey::DatasetHashKey(const std::string &path, int d0, int d1) :
		m_path(path), m_d0(d0), m_d1(d1) {
}

bool DatasetHashKey::operator==(const DatasetHashKey &lhs) const {
	return lhs.m_path == m_path && lhs.m_d0 == m_d0 && lhs.m_d1 == m_d1;
}

std::string DatasetHashKey::hashKey() const {
	std::stringstream ss;
	ss << m_path << "_" << m_d0 << "_" << m_d1;
	return ss.str();
}

DatasetHashKey::DatasetHashKey(const DatasetHashKey &tc) {
	m_path=tc.m_path;
	m_d0=tc.m_d0;
	m_d1=tc.m_d1;
 }

DatasetHashKey & DatasetHashKey::operator= ( const DatasetHashKey & val )
{
	if (this != &val)
	{
		m_path=val.m_path;
		m_d0=val.m_d0;
		m_d1=val.m_d1;
	}
	return *this;
}
void DatasetHashKey::dump()const
{
	std::cout<<m_path<<" "<<m_d0<<" "<<m_d1<<std::endl;
}
