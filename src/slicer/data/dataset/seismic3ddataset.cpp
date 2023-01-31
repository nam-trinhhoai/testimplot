#include "seismic3ddataset.h"
#include "seismic3ddatasetgrahicrepfactory.h"
#include <QFileInfo>
#include <QDebug>
#include <iostream>
#include "Xt.h"
#include "cudaimagepaletteholder.h"
#include "slicerep.h"
#include "slicepositioncontroler.h"
#include <cuda.h>
#include "cuda_volume.h"
#include "cuda_algo.h"
#include <QRunnable>
#include <QThreadPool>
#include <QElapsedTimer>
#include  "cudadatasetminmaxtile.h"
#include "GeotimeProjectManagerWidget.h" // to get file axis
#include "sampletypebinder.h"

#ifndef SAWP
#define SWAP(_a_, _b_, _temp_) { \
	_temp_ = _b_; \
	_b_ = _a_; \
	_a_ = _temp_; }
#endif

static void endianSwap(void *_data, long size, int length)
{
  long i, j;
  unsigned char temp, *data = (unsigned char*)_data;
  
  for (i=0; i<size; i++)
    for (j=0; j<length/2; j++)
      SWAP(data[i*length+j],  data[i*length+(length-1-j)], temp)
}

Seismic3DDataset::Seismic3DDataset(SeismicSurvey *survey,const QString &name,
		WorkingSetManager *workingSet, CUBE_TYPE type, QString idPath, QObject *parent) :
		Seismic3DAbstractDataset(survey,name, workingSet, type, idPath, parent) , m_lock() {
	m_repFactory = new Seismic3DDatasetGraphicRepFactory(this);
	m_currentFile = nullptr;
	m_headerLength = 0;
	// pfileio2 = nullptr;
}

IGraphicRepFactory* Seismic3DDataset::graphicRepFactory() {
	return m_repFactory;
}

void Seismic3DDataset::loadFromXt(const std::string &path, int dimVHint) {
	m_path=path;
	{
		inri::Xt xt(path.c_str());
		if (!xt.is_valid()) {
			std::cerr << "xt cube is not valid (" << path << ")" << std::endl;
			return;
		}
		m_height = xt.nSamples();
		m_width = xt.nRecords();
		m_depth = xt.nSlices();
		m_dimV = xt.pixel_dimensions();
		m_sampleType = translateType(xt.type());

		if (dimVHint>0 && m_dimV==1 && m_height%dimVHint==0) {
			m_height = m_height / dimVHint;
			m_dimV = dimVHint;
		}

		m_seismicAddon.set(
			xt.startSamples(), xt.stepSamples(),
			xt.startRecord(), xt.stepRecords(),
			xt.startSlice(), xt.stepSlices());
		int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(QString::fromStdString(path));
		m_seismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);

		// Skip header
		m_headerLength = (size_t)xt.header_size();
	}

	std::string full_filename = path;
	std::string extension = full_filename.substr(full_filename.find_last_of(".") + 1);
	std::string filename;
	if ( extension == "cwt" )
	{
		size_t lastindex = full_filename.find_last_of("."); 
		filename = full_filename.substr(0, lastindex) + ".xt"; 
	}
	else
	{
		filename = path;
	}
	fprintf(stderr, "filename: %s --> %s\n", path.c_str(), filename.c_str());

	tryInitRangeLock(filename); // if dynamic defined in xt file use it as default range lock

	m_currentFile = fopen(filename.c_str(), "rb");
	if (!m_currentFile) {
		fprintf(stderr, "Error opening file '%s'\n", filename.c_str());
		return;
	}

	// pfileio2 = new FILEIO2();
	// pfileio2->openForRead((char*)path.c_str());


	//initialize here a default transformation
	initializeTransformation();
}

void Seismic3DDataset::loadInlineXLine(CUDAImagePaletteHolder *image,
		SliceDirection dir, unsigned int z, unsigned c, SpectralImageCache* cache) {

	if (dir == SliceDirection::Inline) {

		class InlineRunnable: public QRunnable {
			Seismic3DDataset *m_d;
			unsigned int m_pos;
			CUDAImagePaletteHolder *m_image;
			unsigned int m_c;
			void* m_cache;
		public:
			InlineRunnable(Seismic3DDataset *e, CUDAImagePaletteHolder *image,
					unsigned int z, unsigned int c, void* cache) :
					QRunnable() {
				m_d = e;
				m_pos = z;
				m_image = image;
				m_c = c;
				m_cache = cache;
			}
			void run() { 
				/*std::size_t w = m_d->m_width;
				std::size_t h = m_d->m_height;
				std::size_t d = m_d->m_depth;
				std::vector<short> tmp;
				tmp.resize(w * h * m_d->dimV());
				//short tmp[w * h];
				{
					QMutexLocker locker(&m_d->m_lock);
					size_t absolutePosition = m_d->m_headerLength
							+ w * h * m_pos * sizeof(short);
					fseek(m_d->m_currentFile, absolutePosition, SEEK_SET);
					fread(tmp.data(), sizeof(short), w * h, m_d->m_currentFile);
					// m_d->pfileio2->inlineRead(m_pos, tmp.data());
					// endianSwap(tmp.data(), w*h, sizeof(short));
				}*/
				QList<ChannelCouple> imageAndChannels;
				ChannelCouple couple;
				couple.c = m_c;
				couple.image = m_image;
				imageAndChannels.push_back(couple);
				SampleTypeBinder binder(m_image->sampleType());
				binder.bind<ReadInlineBlockKernel>(m_d, imageAndChannels, m_pos, m_cache);

				// apply range lock
				if (m_d->m_rangeLock) {
					m_image->setRange(m_d->m_lockedRange);
				}
			}
		};
		qDebug()<<"Loading slice:"<<z;
		void* cachePtr = nullptr;
		MonoBlockSpectralImageCache* monoCache = dynamic_cast<MonoBlockSpectralImageCache*>(cache);
		if (monoCache!=nullptr) {
			cachePtr = static_cast<void*>(monoCache->buffer().data());
		}
		InlineRunnable *r = new InlineRunnable(this, image, z, c, cachePtr);
		//QThreadPool::globalInstance()->start(r);
		r->run();
		delete r;
	} else {
		class XlineRunnable: public QRunnable {
			Seismic3DDataset *m_d;
			unsigned int m_pos;
			CUDAImagePaletteHolder *m_image;
			unsigned int m_c;
			void* m_cache;
		public:
			XlineRunnable(Seismic3DDataset *e, CUDAImagePaletteHolder *image,
					unsigned int z, unsigned int c, void* cache) :
					QRunnable() {
				m_d = e;
				m_pos = z;
				m_image = image;
				m_c = c;
				m_cache = cache;
			}
			void run() {
				QList<ChannelCouple> imageAndChannels;
				ChannelCouple couple;
				couple.c = m_c;
				couple.image = m_image;
				imageAndChannels.push_back(couple);
				SampleTypeBinder binder(m_image->sampleType());
				binder.bind<ReadXLineBlockKernel>(m_d, imageAndChannels, m_pos, m_cache);

				// apply range lock
				if (m_d->m_rangeLock) {
					m_image->setRange(m_d->m_lockedRange);
				}
				/*std::size_t w = m_d->m_width;
				std::size_t h = m_d->m_height;
				std::size_t d = m_d->m_depth;
				std::vector<short> temp;
				temp.resize(h * d, 0);
				//short temp[h * d];
				//memset(temp, 0, h * d * sizeof(short));
				{
					QMutexLocker locker(&m_d->m_lock);
					
					fseek(m_d->m_currentFile,
							m_d->m_headerLength + m_pos * h * sizeof(short),
							SEEK_SET);

					size_t seekOffset = (h * w - h) * sizeof(short);
					int numTile = d / TILE_SIZE + 1;
					for (int i = 0; i < numTile; i++) {
						int d0 = i * TILE_SIZE;
						int d1 = d0 + TILE_SIZE;
						if (d1 > d)
							d1 = d;
						for (int k = d0; k < d1; k++) {
							fread(temp.data() + k * h, sizeof(short), h,
									m_d->m_currentFile);
							fseek(m_d->m_currentFile, seekOffset, SEEK_CUR);
						}
						m_image->updateTexture(temp.data(), true);
					}*/
					

				/*
					short *buff = (short*)calloc(w*h, sizeof(short));
					short *p0 = (short*)temp.data();
					for (int iz=0; iz<d; iz++)
					{
						if ( iz%100 == 0 ) fprintf(stderr, "read: %d - %d\n", iz, d);
						m_d->pfileio2->inlineRead(iz, buff);
						endianSwap(buff, w*h, sizeof(short));
						for (int y=0; y<h; y++)
							p0[h*iz+y] = buff[h*m_pos+y];
						m_image->updateTexture(temp.data(), true);
					}
					free(buff);
					*/
				//}

			}
		};
		void* cachePtr = nullptr;
		MonoBlockSpectralImageCache* monoCache = dynamic_cast<MonoBlockSpectralImageCache*>(cache);
		if (monoCache!=nullptr) {
			cachePtr = static_cast<void*>(monoCache->buffer().data());
		}
		XlineRunnable *r = new XlineRunnable(this, image, z, c, cachePtr);
		//QThreadPool::globalInstance()->start(r);
		r->run();
		delete r;
	}
}

void Seismic3DDataset::loadRandomLine(CUDAImagePaletteHolder *cudaImage,
		const QPolygon& randomLine, unsigned int c, SpectralImageCache* cache) {
	class InlineRunnable: public QRunnable {
		Seismic3DDataset *m_d;
		const QPolygon& m_randomLine;
		CUDAImagePaletteHolder *m_image;
		unsigned int m_c;
		void* m_cache;
	public:
		InlineRunnable(Seismic3DDataset *e, CUDAImagePaletteHolder *image,
				const QPolygon& randomLine, unsigned int c, void* cache) :
				QRunnable(), m_randomLine(randomLine) {
			m_d = e;
			m_image = image;
			m_c = c;
			m_cache = cache;
		}
		void run() {
			QList<ChannelCouple> imageAndChannels;
			ChannelCouple couple;
			couple.c = m_c;
			couple.image = m_image;
			imageAndChannels.push_back(couple);

			SampleTypeBinder binder(m_image->sampleType());
			binder.bind<ReadRandomLineKernel>(m_d, imageAndChannels, m_randomLine, m_cache);

			// apply range lock
			if (m_d->m_rangeLock) {
				m_image->setRange(m_d->m_lockedRange);
			}
			/*std::size_t w = m_d->m_width;
			std::size_t h = m_d->m_height;
			std::size_t d = m_d->m_depth;
			std::vector<short> tmp;
			tmp.resize(m_randomLine.size() * h);
			//short tmp[w * h];
			{
				QMutexLocker locker(&m_d->m_lock);
				for (std::size_t idx=0; idx<m_randomLine.size(); idx++) {
					long y = m_randomLine[idx].x();
					long z = m_randomLine[idx].y();
					if (y>=0 && y<w && z>=0 && z<d) {
						fseek(m_d->m_currentFile,
								m_d->m_headerLength + (z * w + y) * h * sizeof(short),
								SEEK_SET);
						fread(tmp.data() + idx * h, sizeof(short), h,
														m_d->m_currentFile);
					}
				}
			}

			m_image->updateTexture(tmp.data(), true);*/
		}
	};
//	qDebug()<<"Loading random line";
	void* cachePtr = nullptr;
	MonoBlockSpectralImageCache* monoCache = dynamic_cast<MonoBlockSpectralImageCache*>(cache);
	if (monoCache!=nullptr) {
		cachePtr = static_cast<void*>(monoCache->buffer().data());
	}
	InlineRunnable *r = new InlineRunnable(this, cudaImage, randomLine, c, cachePtr);
	//QThreadPool::globalInstance()->start(r);
	r->run();
	delete r;
}

// Issue with cube of type InputType == double
template<typename InputType>
struct Seismic3DDataset_InitMinMaxFromTypeKernel {
	static void run(float& min, float& max) {
		max=std::numeric_limits<InputType>::lowest();
		min=std::numeric_limits<InputType>::max();
	}
};

QVector2D Seismic3DDataset::minMax(int channel, bool forced) {
	//By default dynamic is already defined
	if(!forced && m_type==CUBE_TYPE::RGT)
	{
		return QVector2D(0,32000);
	}
	if (m_internalMinMaxCache.initialized && !forced)
		return m_internalMinMaxCache.range;

	CUDADatasetMinMaxTile tile(width(),height(),SEISMIC3DDATASET_TILE_SIZE);
	int numTile = depth() / SEISMIC3DDATASET_TILE_SIZE + 1;
	QVector<QVector2D> tileCoords;
	float min, max;
	SampleTypeBinder binder(m_sampleType);
	binder.bind<Seismic3DDataset_InitMinMaxFromTypeKernel>(min, max);
	//float max=std::numeric_limits<short>::lowest();
	//float min=std::numeric_limits<short>::max();
	for (int i = 0; i < numTile; i++) {
		int d0 = i * SEISMIC3DDATASET_TILE_SIZE;
		int d1 = d0 + SEISMIC3DDATASET_TILE_SIZE;
		if (d1 > depth())
			d1=depth();
		QVector2D temp=tile.minMax(d0,d1,this, channel);
		min=std::min(min,temp.x());
		max=std::max(max,temp.y());
	}
	m_internalMinMaxCache.range=QVector2D(min,max);
	m_internalMinMaxCache.initialized = true;
	std::cout<<min<<"\t"<<max<<std::endl;
	return m_internalMinMaxCache.range;
}

bool Seismic3DDataset::writeRangeToFile(const QVector2D& range) {
	std::string full_filename = m_path;
	std::string extension = full_filename.substr(full_filename.find_last_of(".") + 1);
	std::string filename;
	if ( extension == "cwt" )
	{
		size_t lastindex = full_filename.find_last_of(".");
		filename = full_filename.substr(0, lastindex) + ".xt";
	}
	else
	{
		filename = m_path;
	}
	return Seismic3DAbstractDataset::writeRangeToFile(range, filename);
}


Seismic3DDataset::~Seismic3DDataset() {
	if (m_currentFile)
		fclose(m_currentFile);
}
