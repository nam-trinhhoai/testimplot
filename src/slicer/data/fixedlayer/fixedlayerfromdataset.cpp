#include "fixedlayerfromdataset.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "cpuimagepaletteholder.h"
#include "fixedlayerfromdatasetgraphicrepfactory.h"
#include "textcolortreewidgetitemdecorator.h"
#include <memory>

const QString FixedLayerFromDataset::ISOCHRONE = QString("isochrone");

FixedLayerFromDataset::FixedLayerFromDataset(QString name, WorkingSetManager *workingSet,
		Seismic3DAbstractDataset* dataset, QObject *parent) : IData(workingSet, parent) {
	m_dataset = dataset;
	m_name = name;
	m_color = Qt::blue;
	m_isTemporaryData = false;
	m_repFactory.reset(new FixedLayerFromDatasetGraphicRepFactory(this));

	m_decorator = nullptr;
}

FixedLayerFromDataset::~FixedLayerFromDataset() {
//	std::map<QString, CUDAImagePaletteHolder*>::iterator it = m_images.begin();
//	while ( it!=m_images.end() ) {
//		if ( propName.compare((*it).first) == 0) {
//			delete (*it).second;
//		}
//	}
}

CUDAImagePaletteHolder* FixedLayerFromDataset::image(QString propName) {
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

bool FixedLayerFromDataset::writeProperty(float *tab, QString propName) {
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

bool FixedLayerFromDataset::readProperty(float *tab, QString propName) {
	CUDAImagePaletteHolder* holder = image(propName);
	if (holder) {
		holder->lockPointer();
		memcpy(tab, holder->backingPointer(), sizeof(float)*width()*depth());
		holder->unlockPointer();
	}
	return holder!=nullptr;
}

bool FixedLayerFromDataset::saveProperty(QString filename, QString propName) {
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

bool FixedLayerFromDataset::loadProperty(QString filename, QString propName) {
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

QVector<QString> FixedLayerFromDataset::keys() {
	QVector<QString> output;
	output.resize(m_images.size());
	std::size_t index = 0;
	for (auto& e : m_images) {
		output[index] = e.first;
		index++;
	}
	return output;
}

unsigned int FixedLayerFromDataset::width() const {
	return m_dataset->width();
}

unsigned int FixedLayerFromDataset::depth() const {
	return m_dataset->depth();
}

unsigned int FixedLayerFromDataset::getNbProfiles() const {
	return depth();
}

unsigned int FixedLayerFromDataset::getNbTraces() const {
	return width();
}

float FixedLayerFromDataset::getStepSample() {
	return m_dataset->sampleTransformation()->a();
}

float FixedLayerFromDataset::getOriginSample() {
	return m_dataset->sampleTransformation()->b();
}

//IData
IGraphicRepFactory* FixedLayerFromDataset::graphicRepFactory() {
	return m_repFactory.get();
}

QUuid FixedLayerFromDataset::dataID() const {
	return m_dataset->dataID();
}

QString FixedLayerFromDataset::name() const {
	return m_name;
}

void FixedLayerFromDataset::deleteGraphicItemDataContent(QGraphicsItem* item) {
	std::map<QString, std::shared_ptr<CUDAImagePaletteHolder>>::iterator it = m_images.begin();
	while (it!=m_images.end()) {
		deleteData(it->second.get(), item, -9999);
		it++;
	}

	it = m_images.begin();
	while (it!=m_images.end()) {
		emit propertyModified(it->first);
		it++;
	}
}

IsoSurfaceBuffer FixedLayerFromDataset::getIsoBuffer()
{
	IsoSurfaceBuffer res;

	CUDAImagePaletteHolder* isobuffer = image(ISOCHRONE);
	if(isobuffer==nullptr ) return res;

	res.buffer  = std::make_shared<CPUImagePaletteHolder>(width(), depth(), ImageFormats::QSampleType::FLOAT32,m_dataset->ijToXYTransfo());
	//res.buffer = new CPUImagePaletteHolder(width(), depth(), ImageFormats::QSampleType::FLOAT32,m_dataset->ijToXYTransfo());

	isobuffer->lockPointer();

	void* tab = isobuffer->backingPointer();

	QByteArray array(width()*depth()*sizeof(float),0);

	memcpy(array.data(), tab,width()*depth()*sizeof(float) );


	isobuffer->unlockPointer();

	res.buffer->updateTexture(array, false);


	res.originSample = getOriginSample();
	res.stepSample = getStepSample();

	return res;
}

QColor FixedLayerFromDataset::getColor() const {
	return m_color;
}

void FixedLayerFromDataset::setColor(const QColor& color) {
	if (m_color!=color) {
		m_color = color;

		emit colorChanged(m_color);
	}
}

bool FixedLayerFromDataset::loadColor(const QString& colorFilePath) {
	bool valid;
	QColor color = loadColorFromFile(colorFilePath, &valid);
	if (valid) {
		m_color = color;
	}
	return valid;
}

bool FixedLayerFromDataset::saveColor(const QString& colorFilePath) const {
	return saveColorToFile(colorFilePath, m_color);
}


QColor FixedLayerFromDataset::loadColorFromFile(const QString& colorFilePath, bool* ok) {
	FILE* file = fopen(colorFilePath.toStdString().c_str(), "r");
	bool valid = file != nullptr;
	QColor loadedColor;
	if (valid) {
		char buff[4096];
		fscanf(file, "color file version 1.0\n", buff);

		int matchNumber = fscanf(file, "color: %[^\n]\n", buff);
		valid = matchNumber==1;
		if (valid) {
			loadedColor = QColor(buff);
			valid = loadedColor.isValid();
		}
		fclose(file);
	}
	if (ok!=nullptr) {
		*ok = valid;
	}
	return loadedColor;
}

bool FixedLayerFromDataset::saveColorToFile(const QString& colorFilePath, const QColor& color) {
	FILE* file = fopen(colorFilePath.toStdString().c_str(), "w");
	bool valid = file != nullptr;
	if (valid) {
		QString colorName = color.name();
		QString textToWrite = "color file version 1.0\ncolor: " + colorName + "\n";
		QByteArray colorBuf = textToWrite.toUtf8();
		fwrite(colorBuf.data(), sizeof(char), colorBuf.size(), file);
		fclose(file);
	}
	return valid;
}

bool FixedLayerFromDataset::isTemporaryData() const {
	return m_isTemporaryData;
}

void FixedLayerFromDataset::toggleTemporaryData(bool val) {
	if (val!=m_isTemporaryData) {
		m_isTemporaryData = val;
		emit isTemporaryDataChanged(m_isTemporaryData);

		// update m_decorator, could be done in another function
		if (m_decorator && m_isTemporaryData) {
			m_decorator->setColor(QColor(Qt::cyan));
		} else if (m_decorator) {
			m_decorator->unsetColor();
		}
	}
}

ITreeWidgetItemDecorator* FixedLayerFromDataset::getTreeWidgetItemDecorator() {
	if (m_decorator==nullptr && m_isTemporaryData) {
		m_decorator = new TextColorTreeWidgetItemDecorator(QColor(Qt::cyan), this);
	} else if (m_decorator==nullptr) {
		m_decorator = new TextColorTreeWidgetItemDecorator(this);
	}
	return m_decorator;
}

