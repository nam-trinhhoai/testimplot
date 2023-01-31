#include "datacontroler.h"

DataControler::DataControler(QObject *parent) :
		QObject(parent) {
	m_provider = parent;
}

DataControler::~DataControler() {

}

QObject* DataControler::provider() const {
	return m_provider;
}
