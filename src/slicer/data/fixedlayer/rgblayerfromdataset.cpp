#include "rgblayerfromdataset.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "rgblayerfromdatasetgraphicrepfactory.h"

const QString RgbLayerFromDataset::ISOCHRONE = QString("isochrone");

RgbLayerFromDataset::RgbLayerFromDataset(QString name, WorkingSetManager *workingSet,
		Seismic3DAbstractDataset* dataset, QObject *parent) : IData(workingSet, parent) {
	m_dataset = dataset;
	m_name = name;
	m_repFactory.reset(new RgbLayerFromDatasetGraphicRepFactory(this));
}

RgbLayerFromDataset::~RgbLayerFromDataset() {
//	std::map<QString, CUDAImagePaletteHolder*>::iterator it = m_images.begin();
//	while ( it!=m_images.end() ) {
//		if ( propName.compare((*it).first) == 0) {
//			delete (*it).second;
//		}
//	}
}

CUDAImagePaletteHolder* RgbLayerFromDataset::image(QString propName) {
	CUDAImagePaletteHolder* found = nullptr;

	std::map<QString, std::shared_ptr<CUDAImagePaletteHolder>>::iterator it = m_images.begin();
	while ( it!=m_images.end() && found==nullptr ) {
		if ( propName.compare((*it).first) == 0) {
			found = (*it).second.get();
		}
		it ++;
	}
	return found;
}

bool RgbLayerFromDataset::writeProperty(float *tab, QString propName) {
	CUDAImagePaletteHolder* holder = image(propName);
	bool newHolder = false;
	if (!holder) {
		newHolder = true;
		holder = new CUDAImagePaletteHolder(width(), depth(), ImageFormats::QSampleType::FLOAT32,
				m_dataset->ijToXYTransfo(), parent());
		std::shared_ptr<CUDAImagePaletteHolder> sharedPtr(holder);
		m_images[propName] = sharedPtr;
	}
	holder->lockPointer();
	memcpy(holder->backingPointer(), tab, sizeof(float)*width()*depth());
	holder->unlockPointer();
	if (newHolder) {
		emit newPropertyCreated(propName);
	} else {
		emit propertyModified(propName);
	}
	return true;
}

bool RgbLayerFromDataset::readProperty(float *tab, QString propName) {
	CUDAImagePaletteHolder* holder = image(propName);
	if (holder) {
		holder->lockPointer();
		memcpy(tab, holder->backingPointer(), sizeof(float)*width()*depth());
		holder->unlockPointer();
	}
	return holder!=nullptr;
}

bool RgbLayerFromDataset::saveProperty(QString filename, QString propName) {
	CUDAImagePaletteHolder* holder = image(propName);
	bool valid = holder!=nullptr;
	if (valid) {
		FILE* file = fopen(filename.toStdString().c_str(), "w");
		valid = file != nullptr;
		if (valid) {
			holder->lockPointer();
			fwrite(holder->backingPointer(), sizeof(float), width()*depth(), file);
			holder->unlockPointer();
			fclose(file);
		}
	}
	return valid;
}

bool RgbLayerFromDataset::loadProperty(QString filename, QString propName) {
	CUDAImagePaletteHolder* holder = image(propName);
	bool newHolder = false;
	if (!holder) {
		newHolder  = true;
		holder = new CUDAImagePaletteHolder(width(), depth(), ImageFormats::QSampleType::FLOAT32,
				m_dataset->ijToXYTransfo(), parent());
		std::shared_ptr<CUDAImagePaletteHolder> sharedPtr(holder);
		m_images[propName] = sharedPtr;
	}
	FILE* file = fopen(filename.toStdString().c_str(), "r");
	bool valid = file != nullptr;
	if (valid) {
		holder->lockPointer();
		fread(holder->backingPointer(), sizeof(float), width()*depth(), file);
		holder->unlockPointer();
		fclose(file);
	}
	if (newHolder && valid) {
		emit newPropertyCreated(propName);
	} else {
		emit propertyModified(propName);
	}
	return valid;
}

QVector<QString> RgbLayerFromDataset::keys() {
	QVector<QString> output;
	output.resize(m_images.size());
	std::size_t index = 0;
	for (auto& e : m_images) {
		output[index] = e.first;
		index++;
	}
	return output;
}

unsigned int RgbLayerFromDataset::width() const {
	return m_dataset->width();
}

unsigned int RgbLayerFromDataset::depth() const {
	return m_dataset->depth();
}

unsigned int RgbLayerFromDataset::getNbProfiles() const {
	return depth();
}

unsigned int RgbLayerFromDataset::getNbTraces() const {
	return width();
}

float RgbLayerFromDataset::getStepSample() {
	return m_dataset->sampleTransformation()->a();
}

float RgbLayerFromDataset::getOriginSample() {
	return m_dataset->sampleTransformation()->b();
}

//IData
IGraphicRepFactory* RgbLayerFromDataset::graphicRepFactory() {
	return m_repFactory.get();
}

QUuid RgbLayerFromDataset::dataID() const {
	return m_dataset->dataID();
}

QString RgbLayerFromDataset::name() const {
	return m_name;
}
