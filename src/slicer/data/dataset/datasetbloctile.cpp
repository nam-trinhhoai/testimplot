#include "datasetbloctile.h"
#include <cuda_runtime.h>
#include "cuda_common_helpers.h"
#include "sampletypebinder.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>

DatasetBlocTile::DatasetBlocTile(const std::string &path, int channel, size_t headerLength,
		int w, int h, int d0, int d1, int dimV, ImageFormats::QSampleType sampleType) {
	m_d0 = d0;
	m_d1 = d1;
	m_dimV = dimV;
	m_h = h;
	m_w = w;

	m_depth = m_d1 - m_d0;
	m_path = path;
	m_sampleType = sampleType;
	m_channel = channel;

	//int alignement = boost::iostreams::mapped_file_sink::alignment();
	int alignement=sysconf(_SC_PAGE_SIZE);
	size_t absolutePosition = headerLength + static_cast<std::size_t>(w) * h * m_d0 * m_sampleType.byte_size() * m_dimV;

	m_realOffset = roundDown(absolutePosition, alignement);
	m_delta = absolutePosition - m_realOffset;
	m_blocSize = w * h * m_depth * m_sampleType.byte_size() * m_dimV;

	m_memoryCost = m_blocSize + m_delta;
	m_buffer=nullptr;

	m_fileOpened=false;
}

size_t DatasetBlocTile::memoryCost() {
	return m_memoryCost;
}

DatasetHashKey DatasetBlocTile::key() const{
	return DatasetHashKey(m_path, m_d0, m_d1);
}

size_t DatasetBlocTile::roundDown(size_t numToRound, size_t multiple) {
	if (multiple == 0)
		return numToRound;

	int remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound - remainder;
}

size_t DatasetBlocTile::roundUp(size_t numToRound, size_t multiple) {
	if (multiple == 0)
		return numToRound;

	int remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound + remainder;
}


DatasetBlocTile::~DatasetBlocTile() {
	if (m_fileOpened)
	{
		//munmap (static_cast <void*>(m_buffer), memoryCost());
		free(m_buffer);
	}
}

template<typename InputType>
struct ReorganizeBufferKernel {
	static void run(void* _ori, void* _out, std::size_t N, std::size_t dimV, std::size_t channel) {
		InputType* ori = static_cast<InputType*>(_ori);
		InputType* out = static_cast<InputType*>(_out);
		#pragma omp parallel for
		for (std::size_t idx=0; idx<N; idx++) {
			out[idx] = ori[idx*dimV+channel];
		}
	}
};

void * DatasetBlocTile::buffer() {
	if(!m_fileOpened)
	{
		//https://gist.github.com/sjolsen/6024625
		size_t filesize=memoryCost();
		int fd = open (m_path.c_str (), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
		if (fd == -1)
			throw std::runtime_error (std::strerror (errno) + std::string (" (open)"));

		m_buffer = mmap (nullptr, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m_realOffset);
		close (fd);//we kill the file descriptor
		if (m_buffer == MAP_FAILED)
			throw std::runtime_error (std::strerror (errno) + std::string (" (mmap)"));

		m_fileOpened=true;

		void* tab = malloc(m_w * m_h * m_depth * m_sampleType.byte_size());
		SampleTypeBinder binder(m_sampleType);
		binder.bind<ReorganizeBufferKernel>(m_buffer, tab, m_w * m_h * m_depth, m_dimV, m_channel);

		munmap (static_cast <void*>(m_buffer), memoryCost());
		m_buffer = tab;
	}
	return  m_buffer;
}


