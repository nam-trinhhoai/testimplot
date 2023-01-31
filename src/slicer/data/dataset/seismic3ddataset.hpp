#ifndef SEISMIC3DDATASET_HPP_
#define SEISMIC3DDATASET_HPP_

#include "seismic3ddataset.h"
#include "sampletypebinder.h"
#include "issame.h"
#include "cudaimagepaletteholder.h"
#include <QElapsedTimer>
#include <algorithm>

#define SEISMIC3DDATASET_TILE_SIZE  128 // copied from .cpp file

template<typename InputType>
void Seismic3DDataset_swapValue(InputType& val) {
	char tmp;
	char* it1 = (char*) &val;
	char* it2 = (char*) ((&val)+1);
	it2--;
	while (it1<it2) {
		tmp = *it1;
		*it1 = *it2;
		*it2 = tmp;
		it1++;
		it2--;
	}
}

// should check conversions
template<typename InputType, typename OutputType>
void Seismic3DDataset::swapAndCopyTab(InputType* tabIn, OutputType* tabOut, std::size_t size) {
	if (size>1000000) {
		#pragma omp parallel for
		for (std::size_t i=0; i<size; i++) {
			InputType val = tabIn[i];
			char tmp;
			char* it1 = (char*) &val;
			char* it2 = (char*) ((&val)+1);
			it2--;
			while (it1<it2) {
				tmp = *it1;
				*it1 = *it2;
				*it2 = tmp;
				it1++;
				it2--;
			}
			tabOut[i] = val;
		}
	} else {
		for (std::size_t i=0; i<size; i++) {
			InputType val = tabIn[i];
			char tmp;
			char* it1 = (char*) &val;
			char* it2 = (char*) ((&val)+1);
			it2--;
			while (it1<it2) {
				tmp = *it1;
				*it1 = *it2;
				*it2 = tmp;
				it1++;
				it2--;
			}
			tabOut[i] = val;
		}
	}
}

// should check conversions
template<typename InputType>
void Seismic3DDataset::swapTab(InputType* tabIn, std::size_t size) {
	if (size>1000000) {
		#pragma omp parallel for
		for (std::size_t i=0; i<size; i++) {
			char tmp;
			char* it1 = (char*) (tabIn+i);
			char* it2 = (char*) (tabIn+i+1);
			it2--;
			while (it1<it2) {
				tmp = *it1;
				*it1 = *it2;
				*it2 = tmp;
				it1++;
				it2--;
			}
		}
	} else {
		for (std::size_t i=0; i<size; i++) {
			char tmp;
			char* it1 = (char*) (tabIn+i);
			char* it2 = (char*) (tabIn+i+1);
			it2--;
			while (it1<it2) {
				tmp = *it1;
				*it1 = *it2;
				*it2 = tmp;
				it1++;
				it2--;
			}
		}
	}
}

template<typename OutputType>
class ReadFromFileKernelHolder {
public:
	static void readFile(FILE* file, OutputType* tab, std::size_t size,
			ImageFormats::QSampleType inputType, bool returnBigEndian=true) {
		SampleTypeBinder binder(inputType);
		binder.bind<ReadFromFileKernel>(file, tab, size, returnBigEndian);
	}
private:
	template<typename InputType>
	struct ReadFromFileKernel {
		static void run(FILE* file, OutputType* tab, std::size_t size, bool returnBigEndian=true) {
			InputType* buf;
			std::vector<InputType> _buf;
			if (isSameType<InputType, OutputType>::value) {
				buf = (InputType*) ((void*)tab);
			} else {
				_buf.resize(size);
				buf = _buf.data();
			}

			fread(buf, sizeof(InputType), size, file);

			if (!isSameType<InputType, OutputType>::value) {
				Seismic3DDataset::swapAndCopyTab(buf, tab, size);
				if (returnBigEndian) {
					Seismic3DDataset::swapTab(tab, size); // swap back for endianness
				}
			} else if (isSameType<InputType, OutputType>::value&&!returnBigEndian) {
				Seismic3DDataset::swapTab(tab, size); // swap back for endianness
			}
		}
	};
};

template<typename DataType>
void Seismic3DDataset::readInlineBlock(DataType * output,int z0, int z1, bool returnBigEndian) const
{
	QMutexLocker locker(&m_lock);
	std::size_t w = m_width;
	std::size_t h = m_height;

	QElapsedTimer timer;
	size_t absolutePosition = m_headerLength
								+ m_dimV * w * h * z0 * m_sampleType.byte_size();
	fseek(m_currentFile, absolutePosition, SEEK_SET);

	ReadFromFileKernelHolder<DataType>::readFile(m_currentFile, output,
			m_dimV * w * h * (z1-z0), m_sampleType, returnBigEndian);

}

template<typename DataType>
void Seismic3DDataset::readTraceBlock(DataType *output, int y0, int y1, int z, bool returnBigEndian) const {
	QMutexLocker locker(&m_lock);
	std::size_t w = m_width;
	std::size_t h = m_height;

	QElapsedTimer timer;
	size_t absolutePosition = m_headerLength
								+ m_dimV * h * (w * z + y0) * m_sampleType.byte_size();
	fseek(m_currentFile, absolutePosition, SEEK_SET);

	ReadFromFileKernelHolder<DataType>::readFile(m_currentFile, output,
			m_dimV * h * (y1-y0), m_sampleType, returnBigEndian);
}

template<typename DataType>
void Seismic3DDataset::readSubTrace(DataType *output, int x0, int x1, int y, int z, bool returnBigEndian) const {
	QMutexLocker locker(&m_lock);
	std::size_t w = m_width;
	std::size_t h = m_height;

	QElapsedTimer timer;
	size_t absolutePosition = m_headerLength
								+ m_dimV * (h * (w * z + y) + x0) * m_sampleType.byte_size();
	fseek(m_currentFile, absolutePosition, SEEK_SET);

	ReadFromFileKernelHolder<DataType>::readFile(m_currentFile, output,
			m_dimV * (x1-x0), m_sampleType, returnBigEndian);
}

template<typename DataType>
void Seismic3DDataset::readTraceBlockAndSwap(DataType *output, int y0, int y1, int z) const {
	readTraceBlock(output, y0, y1, z, false);
}

template<typename DataType>
void Seismic3DDataset::readSubTraceAndSwap(DataType *output, int x0, int x1, int y, int z) const {
	readSubTrace(output, x0, x1, y, z, false);
}

template <typename DataType>
void Seismic3DDataset::ReadInlineBlockKernel<DataType>::run(const Seismic3DDataset* obj, QList<ChannelCouple> imageAndChannels, int z, void* _cache) {
	std::size_t w = obj->m_width;
	std::size_t h = obj->m_height;
	std::size_t d = obj->m_depth;
	std::size_t dimV = obj->m_dimV;
	std::vector<DataType> tmp;
	tmp.resize(w * h * dimV);
	DataType* cache = (DataType*)_cache;
	//QMutexLocker locker(&obj->m_lock);
	obj->readInlineBlock(tmp.data(), z, z+1);

	if (cache==nullptr) {
		std::size_t N = w * h;
		std::vector<DataType> buf;
		buf.resize(N);
		for (ChannelCouple couple : imageAndChannels) {
			if (couple.c<dimV && couple.c>=0) {
				#pragma omp parallel for
				for (std::size_t idx=0; idx<N; idx++) {
					buf[idx] = tmp[idx*dimV+couple.c];
				}
			} else {
				std::fill(buf.begin(), buf.end(), 0);
			}
			couple.image->updateTexture(buf.data(), true);
		}
	} else {
		#pragma omp parallel for
		for (std::size_t i=0; i<w; i++) {
			for (std::size_t j=0; j<h; j++) {
				std::size_t inIdx = (j + i*h) * dimV;
				std::size_t outIdx = i + j * w;
				for (std::size_t k=0; k<dimV; k++) {
					cache[outIdx + k * w * h] = tmp[inIdx + k];
					Seismic3DDataset_swapValue(cache[outIdx + k * w * h]);
				}
			}
		}
		for (ChannelCouple couple : imageAndChannels) {
			couple.image->updateTexture(cache + couple.c * w * h, false);
		}
	}
}

template <typename DataType>
void Seismic3DDataset::ReadXLineBlockKernel<DataType>::run(const Seismic3DDataset* obj, QList<ChannelCouple> imageAndChannels, int y, void* _cache) {
	std::size_t w = obj->m_width;
	std::size_t h = obj->m_height;
	std::size_t d = obj->m_depth;
	std::size_t dimV = obj->m_dimV;
	std::vector<DataType> temp;
	temp.resize(h * dimV * SEISMIC3DDATASET_TILE_SIZE, 0);
	DataType* cache = (DataType*)_cache;
	//short temp[h * d];
	//memset(temp, 0, h * d * sizeof(short));

	//QMutexLocker locker(&obj->m_lock);

	/*fseek(m_d->m_currentFile,
			m_d->m_headerLength + m_pos * h * sizeof(short),
			SEEK_SET);*/

	std::vector<std::vector<DataType>> bufs;
	if (cache==nullptr) {
		bufs.resize(imageAndChannels.size());

		std::size_t N = d * h;
		for (std::size_t idx=0; idx<imageAndChannels.size(); idx++) {
			bufs[idx].resize(N, 0);
		}
	}
	size_t seekOffset = (h * w - h) * sizeof(DataType);
	int numTile = d / SEISMIC3DDATASET_TILE_SIZE + 1;
	for (int i = 0; i < numTile; i++) {
		int d0 = i * SEISMIC3DDATASET_TILE_SIZE;
		int d1 = d0 + SEISMIC3DDATASET_TILE_SIZE;
		if (d1 > d)
			d1 = d;
		for (int k = d0; k < d1; k++) {
			obj->readTraceBlock(temp.data() + (k-d0)*h * dimV, y, y+1, k);
			/*fread(temp.data() + k * h, sizeof(short), h,
					m_d->m_currentFile);
			fseek(m_d->m_currentFile, seekOffset, SEEK_CUR);*/
		}

		if (cache==nullptr) {
			for (std::size_t idx=0; idx<bufs.size(); idx++) {
				ChannelCouple couple = imageAndChannels[idx];
				std::vector<DataType>& buf = bufs[idx];
				if (couple.c<dimV && couple.c>=0) {
					//#pragma omp parallel for
					for (std::size_t idxTemp=0; idxTemp<h*(d1-d0); idxTemp++) {
						buf[idxTemp + d0 * h] = temp[idxTemp*dimV+couple.c];
					}
				}
				//couple.image->updateTexture(buf.data(), true);
			}
		} else {
			#pragma omp parallel for
			for (std::size_t i=d0; i<d1; i++) {
				for (std::size_t j=0; j<h; j++) {
					std::size_t inIdx = (j + (i-d0)*h) * dimV;
					std::size_t outIdx = i + j * d;
					for (std::size_t k=0; k<dimV; k++) {
						cache[outIdx + k * d * h] = temp[inIdx + k];
						Seismic3DDataset_swapValue(cache[outIdx + k * d * h]);
					}
				}
			}
		}
		//image->updateTexture(temp.data(), true);
	}
	if (cache==nullptr) {
		for(std::size_t idx=0; idx<bufs.size(); idx++) {
			ChannelCouple couple = imageAndChannels[idx];
			std::vector<DataType>& buf = bufs[idx];
			couple.image->updateTexture(buf.data(), true);
		}
	} else {
		for (ChannelCouple couple : imageAndChannels) {
			couple.image->updateTexture(cache + couple.c * d * h, false);
		}
	}
}

template<typename DataType>
void Seismic3DDataset::ReadRandomLineKernel<DataType>::run(const Seismic3DDataset* obj,
		QList<ChannelCouple> imageAndChannels, const QPolygon& poly, void* _cache) {
	std::size_t w = obj->m_width;
	std::size_t h = obj->m_height;
	std::size_t d = obj->m_depth;
	std::size_t dimV = obj->m_dimV;
	std::vector<DataType> tmp;
	tmp.resize(poly.size() * h * dimV);
	DataType* cache = (DataType*)_cache;
	//short tmp[w * h];
	for (std::size_t idx=0; idx<poly.size(); idx++) {
		long y = poly[idx].x();
		long z = poly[idx].y();
		if (y>=0 && y<w && z>=0 && z<d) {
			/*fseek(m_d->m_currentFile,
					m_d->m_headerLength + (z * w + y) * h * sizeof(short),
					SEEK_SET);
			fread(tmp.data() + idx * h, sizeof(short), h,
											m_d->m_currentFile);*/
			obj->readTraceBlock(tmp.data() + idx * h * dimV, y, y+1, z);
		}
	}

	if (cache==nullptr) {
		std::size_t N = poly.size() * h;
		std::vector<DataType> buf;
		buf.resize(N);
		for (ChannelCouple couple : imageAndChannels) {
			if (couple.c<dimV && couple.c>=0) {
				#pragma omp parallel for
				for (std::size_t idx=0; idx<N; idx++) {
					buf[idx] = tmp[idx*dimV+couple.c];
				}
			} else {
				std::fill(buf.begin(), buf.end(), 0);
			}
			couple.image->updateTexture(buf.data(), true);
		}
	} else {
		std::size_t polySize = poly.size();
		#pragma omp parallel for
		for (std::size_t i=0; i<polySize; i++) {
			for (std::size_t j=0; j<h; j++) {
				std::size_t inIdx = (j + i*h) * dimV;
				std::size_t outIdx = i + j * polySize;
				for (std::size_t k=0; k<dimV; k++) {
					cache[outIdx + k * polySize * h] = tmp[inIdx + k];
					Seismic3DDataset_swapValue(cache[outIdx + k * w * h]);
				}
			}
		}
		for (ChannelCouple couple : imageAndChannels) {
			couple.image->updateTexture(cache + couple.c * polySize * h, false);
		}
	}
}

#endif
