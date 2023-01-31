#ifndef PickingInfo_H
#define PickingInfo_H

#include <QUuid>
#include <vector>

class PickingInfo {
public:
	PickingInfo(const QUuid & dataUUID, const std::vector<double>& value) {
		m_dataUUID=dataUUID;
		for(double v:value)
			m_values.push_back(v);
	}
	PickingInfo(const PickingInfo &par) {
		m_dataUUID=par.m_dataUUID;
		for(double v:par.m_values)
			m_values.push_back(v);
	}
	PickingInfo& operator=(const PickingInfo &par) {
		if (this != &par) {
			this->m_dataUUID = par.m_dataUUID;
			for(double v:par.m_values)
				m_values.push_back(v);
		}
		return *this;
	}

	~PickingInfo()
	{}

	QUuid uuid() const {
		return m_dataUUID;
	}
	std::vector<double> value() const {
		return m_values;
	}
private:
	QUuid m_dataUUID;
	std::vector<double> m_values;
};

#endif
