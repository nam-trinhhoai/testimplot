#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "seismicsurvey.h"
#include "gdal.h"
#include "colortableregistry.h"
#include "cudaimagepaletteholder.h"
#include "icontreewidgetitemdecorator.h"

#include <QPolygon>
#include <QDebug>
#include <QFileInfo>

extern "C" {
#define Linux 1
#include "image.h"
#include "comOCR.h"
int iputhline(struct image *nf, char *key, char *buf);
}



SpectralImageCache::SpectralImageCache() {}

SpectralImageCache::~SpectralImageCache() {}

MonoBlockSpectralImageCache::MonoBlockSpectralImageCache(unsigned int width, unsigned int height,
		unsigned int dimV, ImageFormats::QSampleType sampleType) {
	m_width = width;
	m_height = height;
	m_dimV = dimV;
	m_sampleType = sampleType;
	m_buffer.resize(m_width*m_height*m_dimV*m_sampleType.byte_size());
}

MonoBlockSpectralImageCache::~MonoBlockSpectralImageCache() {

}

unsigned int MonoBlockSpectralImageCache::width() const {
	return m_width;
}

unsigned int MonoBlockSpectralImageCache::height() const {
	return m_height;
}

unsigned int MonoBlockSpectralImageCache::dimV() const {
	return m_dimV;
}

ImageFormats::QSampleType MonoBlockSpectralImageCache::sampleType() const {
	return m_sampleType;
}

// fastest axis are width, height and dimV
std::vector<char>& MonoBlockSpectralImageCache::buffer() {
	return m_buffer;
}

bool MonoBlockSpectralImageCache::copy(CUDAImagePaletteHolder *cudaImage, int channel) {
	bool ok = channel>=0 && channel<m_dimV && m_width == cudaImage->width() &&
			m_height == cudaImage->height() && m_sampleType == cudaImage->sampleType();
	if (ok) {
		cudaImage->updateTexture(m_buffer.data() + ((std::size_t)channel) * m_width * m_height * m_sampleType.byte_size(), false);
	}
	return ok;
}

ArraySpectralImageCache::ArraySpectralImageCache(unsigned int width, unsigned int height,
		unsigned int dimV, ImageFormats::QSampleType sampleType) {
	m_width = width;
	m_height = height;
	m_dimV = dimV;
	m_sampleType = sampleType;
	m_buffer.resize(m_dimV);
	for (std::size_t i=0; i<m_dimV; i++) {
		m_buffer[i].resize(m_width*m_height*m_sampleType.byte_size());
	}
}

ArraySpectralImageCache::~ArraySpectralImageCache() {

}

unsigned int ArraySpectralImageCache::width() const {
	return m_width;
}

unsigned int ArraySpectralImageCache::height() const {
	return m_height;
}

unsigned int ArraySpectralImageCache::dimV() const {
	return m_dimV;
}

ImageFormats::QSampleType ArraySpectralImageCache::sampleType() const {
	return m_sampleType;
}

// first vector manage dimV, second width and height; fastest being width
// buffer ready to be copied into cuda image palette holder without transpose
std::vector<std::vector<char>>& ArraySpectralImageCache::buffer() {
	return m_buffer;
}

bool ArraySpectralImageCache::copy(CUDAImagePaletteHolder *cudaImage, int channel) {
	bool ok = channel>=0 && channel<m_dimV && m_width == cudaImage->width() &&
			m_height == cudaImage->height() && m_sampleType == cudaImage->sampleType();
	if (ok) {
		cudaImage->updateTexture(m_buffer[channel].data(), false);
	}
	return ok;
}


Volume::Volume(WorkingSetManager *workingSet, QObject *parent) : IData(workingSet, parent) {

}

Volume::~Volume() {

}

bool Volume::isCompatible(Volume* other) {
	return this->cubeSeismicAddon().compare3DGrid((other->cubeSeismicAddon())) && this->width()==other->width() &&
						this->height()==other->height() && this->depth()==other->depth();
}

Seismic3DAbstractDataset::Seismic3DAbstractDataset(SeismicSurvey *survey,
		const QString &name, WorkingSetManager *workingSet, CUBE_TYPE type,
		QString idPath, QObject *parent) :
		Volume(workingSet, parent), IFileBasedData(idPath) {
	m_name = name;
	m_width = 0;
	m_height = 0;
	m_depth = 0;
	m_type = type;
	m_uuid = QUuid::createUuid();
	m_survey = survey;
	m_internalMinMaxCache.initialized = false;
	m_ijToInlineXline = m_ijToInlineXlineForInline = m_ijToInlineXlineForXline =
			m_ijToXY = nullptr;
	m_TreeDeletionProcess = false;
	m_rangeLock = false;
	m_treeWidgetItemDecorator = nullptr;
}
LookupTable Seismic3DAbstractDataset::defaultLookupTable() const {
	if (type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT)
		return ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
				"Iles-2");
	else
		return ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
				"White-Black");
}
void Seismic3DAbstractDataset::initializeTransformation() {
	m_ijToInlineXline = new Affine2DTransformation(m_width, m_depth, this);

	m_ijToInlineXlineForInline = new Affine2DTransformation(m_width, m_height,
			this);
	m_ijToInlineXlineForXline = new Affine2DTransformation(m_depth, m_height,
			this);

	m_sampleTransformation = new AffineTransformation(this);

	updateIJToXYTransfo();
}

void Seismic3DAbstractDataset::updateIJToXYTransfo() {
	if (m_ijToXY != nullptr)
		m_ijToXY->deleteLater();

	std::array<double, 6> inlineXlineTransfo =
			m_survey->inlineXlineToXYTransfo()->direct();
	std::array<double, 6> ijToInlineXline = m_ijToInlineXline->direct();

	std::array<double, 6> res;
	GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
			res.data());
	m_ijToXY = new Affine2DTransformation(m_ijToInlineXline->width(),
			m_ijToInlineXline->height(), res, this);
}

const Affine2DTransformation* const Seismic3DAbstractDataset::ijToXYTransfo() const {
	return m_ijToXY;
}

QRectF Seismic3DAbstractDataset::inlineXlineExtent() const {
	return m_ijToInlineXline->worldExtent();
}

QVector2D Seismic3DAbstractDataset::minMax(int channel, bool forced) {
	if (m_type == CUBE_TYPE::RGT) {
		return QVector2D(0, 32000);
	}
	return QVector2D(-32000, 32000);
}

QUuid Seismic3DAbstractDataset::dataID() const {
	return m_uuid;
}

Seismic3DAbstractDataset::~Seismic3DAbstractDataset() {
}
const AffineTransformation* const Seismic3DAbstractDataset::sampleTransformation() const {
	return m_sampleTransformation;
}

void Seismic3DAbstractDataset::setSampleTransformation(
		const AffineTransformation &transfo) {
	m_sampleTransformation->deleteLater();
	m_sampleTransformation = new AffineTransformation(transfo);
	m_sampleTransformation->setParent(this);
}

const Affine2DTransformation* const Seismic3DAbstractDataset::ijToInlineXlineTransfo() const {
	return m_ijToInlineXline;
}
void Seismic3DAbstractDataset::setIJToInlineXlineTransfo(
		const Affine2DTransformation &transfo) {
	m_ijToInlineXline->deleteLater();
	m_ijToInlineXline = new Affine2DTransformation(transfo);
	m_ijToInlineXline->setParent(this);
	updateIJToXYTransfo();
}
const Affine2DTransformation* const Seismic3DAbstractDataset::ijToInlineXlineTransfoForInline() const {
	return m_ijToInlineXlineForInline;
}
void Seismic3DAbstractDataset::setIJToInlineXlineTransfoForInline(
		const Affine2DTransformation &transfo) {
	m_ijToInlineXlineForInline->deleteLater();
	m_ijToInlineXlineForInline = new Affine2DTransformation(transfo);
	m_ijToInlineXlineForInline->setParent(this);
}

const Affine2DTransformation* const Seismic3DAbstractDataset::ijToInlineXlineTransfoForXline() const {
	return m_ijToInlineXlineForXline;
}

void Seismic3DAbstractDataset::deleteRep(){
	if(m_TreeDeletionProcess == false){
      m_TreeDeletionProcess = true;
	  emit deletedMenu();
	}
}

void Seismic3DAbstractDataset::setTreeDeletionProcess(bool value){
	m_TreeDeletionProcess = value;
}

void Seismic3DAbstractDataset::addRep(AbstractGraphicRep *pRep){
	if(pRep != nullptr)
		m_RepList.push_back(pRep);
}

void Seismic3DAbstractDataset::deleteRep(AbstractGraphicRep *pRep){
	if(pRep != nullptr)
		m_RepList.removeAll(pRep);
}

int Seismic3DAbstractDataset::getRepListSize(){
	return m_RepList.size();
}

void Seismic3DAbstractDataset::setIJToInlineXlineTransfoForXline(
		const Affine2DTransformation &transfo) {
	m_ijToInlineXlineForXline->deleteLater();
	m_ijToInlineXlineForXline = new Affine2DTransformation(transfo);
	m_ijToInlineXlineForXline->setParent(this);
}

CubeSeismicAddon Seismic3DAbstractDataset::cubeSeismicAddon() const {
	return m_seismicAddon;
}

MonoBlockSpectralImageCache* Seismic3DAbstractDataset::createInlineXLineCache(SliceDirection dir) const {
	MonoBlockSpectralImageCache* out;
	if (dir==SliceDirection::Inline) {
		out = new MonoBlockSpectralImageCache(m_width, m_height, m_dimV, m_sampleType);
	} else {
		out = new MonoBlockSpectralImageCache(m_depth, m_height, m_dimV, m_sampleType);
	}
	return out;
}

MonoBlockSpectralImageCache* Seismic3DAbstractDataset::createRandomCache(const QPolygon& poly) const {
	return new MonoBlockSpectralImageCache(poly.size(), m_height, m_dimV, m_sampleType);
}

ImageFormats::QSampleType Seismic3DAbstractDataset::translateType(const inri::Xt::Type& type) {
        switch(type) {
        case inri::Xt::Signed_8:
                return ImageFormats::QSampleType::INT8;
        case inri::Xt::Unsigned_8:
                return ImageFormats::QSampleType::UINT8;
        case inri::Xt::Signed_16:
                return ImageFormats::QSampleType::INT16;
        case inri::Xt::Unsigned_16:
                return ImageFormats::QSampleType::UINT16;
        case inri::Xt::Signed_32:
                return ImageFormats::QSampleType::INT32;
        case inri::Xt::Unsigned_32:
                return ImageFormats::QSampleType::UINT32;
        case inri::Xt::Signed_64:
                return ImageFormats::QSampleType::INT64;
        case inri::Xt::Unsigned_64:
                return ImageFormats::QSampleType::UINT64;
        case inri::Xt::Float:
                return ImageFormats::QSampleType::FLOAT32;
        case inri::Xt::Double:
                return ImageFormats::QSampleType::FLOAT64;
        case inri::Xt::Unknown:
                return ImageFormats::QSampleType::ERR;
        }
        return ImageFormats::QSampleType::ERR;
}

bool Seismic3DAbstractDataset::isRangeLocked() const {
	return m_rangeLock;
}

const QVector2D& Seismic3DAbstractDataset::lockedRange() const {
	return m_lockedRange;
}

void Seismic3DAbstractDataset::lockRange(const QVector2D& range) {
	if ((range!=m_lockedRange || !m_rangeLock) && range.x()<range.y()) {
		m_lockedRange = range;
		m_rangeLock = true;
		emit rangeLockChanged();
	}
}

void Seismic3DAbstractDataset::unlockRange() {
	if (m_rangeLock) {
		m_rangeLock = false;
		emit rangeLockChanged();
	}
}

void Seismic3DAbstractDataset::tryInitRangeLock(const std::string& xtFile) {
	// get dynamic
	bool minSet = false;
	bool maxSet = false;
	float min=0, max = 1;

	FILE *pFile = fopen(xtFile.c_str(), "r");
	if ( pFile == NULL ) return;
	char str[10000];
	fseek(pFile, 0x4c, SEEK_SET);
	int n = 0, cont = 1;
	while ( cont )
	{
		int nbre = fscanf(pFile, "VMIN=\t%f\n", &min);
		if ( nbre > 0 ) {
			cont = 0;
			minSet = true;
		} else
			fgets(str, 10000, pFile);
		n++;
		if ( n > 40 )
		{
			cont = 0;
			strcpy(str, "Other");
		}
	}
	fseek(pFile, 0x4c, SEEK_SET);
	n = 0, cont = 1;
	while ( cont )
	{
		int nbre = fscanf(pFile, "VMAX=\t%f\n", &max);
		if ( nbre > 0 ) {
			cont = 0;
			maxSet = true;
		} else
			fgets(str, 10000, pFile);
		n++;
		if ( n > 40 )
		{
			cont = 0;
			strcpy(str, "Other");
		}
	}
	fclose(pFile);

	if (minSet && maxSet) {
		lockRange(QVector2D(min, max));
	}
}

bool Seismic3DAbstractDataset::writeRangeToFile(const QVector2D& range, const std::string& xtFile) {
	/* 15/10/2021 : AS
	 * There may be some issues with irephline
	 *
	 * Do not know enough inrimage to manage case VMIN/VMAX not defined in xt comments.
	 * iputhline may help for that case
	 */
	QFileInfo fileInfo(QString::fromStdString(xtFile));
	if (!fileInfo.isWritable()) {
		return false;
	}

	struct stat buffer;
	struct nf_fmt ifmt;
	struct image*  im = image_(const_cast<char*>(xtFile.c_str()), "s", " ", &ifmt);

	QString minStr = QString::number(range.x());
	std::vector<char> valueMinBuf;
	valueMinBuf.resize(minStr.count()+1);
	valueMinBuf[minStr.count()] = 0;
	memcpy(valueMinBuf.data(), minStr.toStdString().c_str(), minStr.count());
	int xx = irephline(im, "VMIN=", 1, valueMinBuf.data());

	QString maxStr = QString::number(range.y());
	std::vector<char> valueMaxBuf;
	valueMaxBuf.resize(maxStr.count()+1);
	valueMaxBuf[maxStr.count()] = 0;
	memcpy(valueMaxBuf.data(), maxStr.toStdString().c_str(), maxStr.count());
	xx = irephline(im, "VMAX=", 1, valueMaxBuf.data());
	c_fermnf(im);

	return true;
}

ITreeWidgetItemDecorator* Seismic3DAbstractDataset::getTreeWidgetItemDecorator() {
	if (m_treeWidgetItemDecorator==nullptr) {
		bool typeFound = true;
		QString typeName;
		if (m_sampleType.bit_size()==8) {
			typeName = "Clair";
		} else if (m_sampleType.bit_size()==16) {
			typeName = "Vif";
		} else if (m_sampleType.bit_size()==32) {
			typeName = "Foncé";
		} else {
			typeFound = false;
		}

		bool unitFound = true;
		QString unitName;
		if (m_seismicAddon.getSampleUnit()==SampleUnit::DEPTH) {
			unitName = "Rouge";
		} else if (m_seismicAddon.getSampleUnit()==SampleUnit::TIME) {
			unitName = "Bleu";
		} else {
			unitFound = false;
		}

		QString cubeTypeName;
		if (m_type==CUBE_TYPE::RGT) {
			cubeTypeName = ".Vert";
		} else if (m_type==CUBE_TYPE::Patch) {
			cubeTypeName = ".Jaune";
		}

		bool iconFound = typeFound && unitFound;
		if (iconFound) {
			QString iconFile = ":/slicer/icons/dataset_icons/" + unitName + typeName + cubeTypeName + ".svg";
			QIcon icon(iconFile);
			m_treeWidgetItemDecorator = new IconTreeWidgetItemDecorator(icon, this);
		}
	}
	return m_treeWidgetItemDecorator;
}
