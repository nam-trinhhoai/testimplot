/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */


#include <QTableView>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QPainter>
#include <QChart>
#include <QLineEdit>
#include <QToolButton>
#include <QLineSeries>
#include <QScatterSeries>
#include <QtCharts>
#include <QRandomGenerator>
#include <QTimer>
#include <QInputDialog>

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include "GeotimeConfiguratorExpertWidget.h"
#include "GeotimeConfiguratorWidget.h"
#include "globalconfig.h"
#include "Xt.h"
#include "imageformats.h"
#include "sampletypebinder.h"
#include "seismic3dabstractdataset.h"
#include "interpolation.h"


// #define __LINUX__
#define EXPORT_LIB __attribute__((visibility("default")))
#include <config.h>
#include <normal.h>
#include <util.h>
#include <cuda_utils.h>
#include <fileio.h>
#include <fileio2.h>
#include <surface_stack.h>
#include "FileConvertionXTCWT.h"
#include <ihm.h>

#include <file_convert.h>

using namespace std;

#define DEBUG0 fprintf(stderr, "%s %d\n", __FILE__, __LINE__);

extern "C" {
#define Linux 1
#include "image.h"
#include "comOCR.h"
int iputhline(struct image *nf, char *key, char *buf);
}

MyThread_file_convertion::MyThread_file_convertion()
{
	this->cpt = 0;
	this->cpt_max = 1;
	this->complete = 0;
	this->abort = 0;
	this->idxfile = 0;
	this->idxfilemax = 1;
}

MyThread_XTCWT_file_convertion::MyThread_XTCWT_file_convertion() : MyThread_file_convertion() {

}

void MyThread_XTCWT_file_convertion::run()
{
	if ( src_filenames.size() == 0 || dst_filenames.size() == 0 ) return;
	cont0 = 1;
	idxfile = 0;
	this->idxfilemax = src_filenames.size();
	while ( cont0 )
	{
		QString src = src_filenames[idxfile];
		QString dst = dst_filenames[idxfile];

		FILEIO2 *p = new FILEIO2();
    	p->createNew((char*)src.toStdString().c_str() , (char*)dst.toStdString().c_str());
		fprintf(stderr, ">>> %s %s\n", src.toStdString().c_str(), (char*)dst.toStdString().c_str());
		int dimx = p->get_dimx();
    	int dimy = p->get_dimy();
    	int dimz = p->get_dimz();
    	int format = p->get_format();
    	int sizeof_ = p->get_sizeof();
    	delete p;	
		FILEIO2 *pdst = new FILEIO2();
    	pdst->openForWrite((char*)dst.toStdString().c_str(), this->cwt_error[idxfile]);
		FILEIO2 *psrc = new FILEIO2();
    	psrc->openForRead((char*)src.toStdString().c_str());	
		
		void *datax = (void*)calloc(dimx*dimy, sizeof_);
		this->cpt_max = dimz-1;
		this->cont = 1;
		this->abort == 0;
		this->current_filename = tiny_filenames[idxfile];
		int z = 0;
    	// or (int z=0; z<dimz; z++)
		while ( cont )
    	{
			this->cpt = z;
        	psrc->inlineRead(z, datax);
        	pdst->inlineWrite(z, datax);
			z++; if ( z >= dimz ) cont = 0;
			if ( this->abort == 1 ) cont = 0;
    	}
		this->cpt = 0;
    	delete pdst;
    	delete psrc;
		free(datax);

		idxfile++;
		if ( idxfile >= idxfilemax ) cont0 = 0;
		if ( this->abort == 1 ) cont0 = 0;
	}
	complete = 1;
}

void copyChar(char* buf, int n, const std::string& str) {
        const char* strBuf = str.c_str();
        int i=0;
        while(i<n-1 && strBuf[i]!=0) {
                buf[i] = strBuf[i];
                i++;
        }
        buf[i]=0;
}

template<typename InputType>
struct CopyToShortKernel {
	static void run(MyThread_FloatShort_file_convertion* pthread, const QString& src, const QString& dst, std::size_t dimX,
			std::size_t dimY, std::size_t dimZ, std::size_t hIn, std::size_t hOut) {
		FILE* fr = fopen(src.toStdString().c_str(), "r");
		FILE* fo = fopen(dst.toStdString().c_str(), "r+");

		double oriMax = 0;
		short outMax = 32000; // arbirary value

		std::size_t N = dimX*dimY;
		std::vector<InputType> inBuffer;
		std::vector<short> outBuffer;
		inBuffer.resize(N);
		outBuffer.resize(N);
		pthread->cpt_max = dimZ-1;
		pthread->cont = 1;
		long z=0;

		pthread->cpt = 0;

		// detect oriMax
		long zStep = dimZ / 4;
		long idxStep = 10;
		while(pthread->cont && z<dimZ) {
			fseek(fr, hIn + z*N*4, SEEK_SET);
			fread(inBuffer.data(), N, 4, fr);
			#pragma omp parallel for
			for (std::size_t idx=0; idx<N; idx+=idxStep) {
				InputType oriVal = inBuffer[idx];
				char tmp;
				char* beginPtr = static_cast<char*>(static_cast<void*>(&oriVal));
				char* endPtr = static_cast<char*>(static_cast<void*>((&oriVal)+1))-1;
				while (beginPtr<endPtr) {
					tmp = *endPtr;
					*endPtr = *beginPtr;
					*beginPtr = tmp;
					beginPtr++;
					endPtr--;
				}
				if (std::fabs(oriVal)>oriMax) {
					oriMax = std::fabs(oriVal);
				}
			}

			z += zStep;
			if ( pthread->abort == 1 ) pthread->cont = 0;
		}
		oriMax = oriMax * 1.1;

		z = 0;

		// apply convertion
		while (pthread->cont) {
			pthread->cpt = z;

			fseek(fr, hIn + z*N*4, SEEK_SET);
			fread(inBuffer.data(), N, 4, fr);

			#pragma omp parallel for
			for (std::size_t idx=0; idx<N; idx++) {
				InputType oriVal = inBuffer[idx];
				char tmp;
				char* beginPtr = static_cast<char*>(static_cast<void*>(&oriVal));
				char* endPtr = static_cast<char*>(static_cast<void*>((&oriVal)+1))-1;
				while (beginPtr<endPtr) {
					tmp = *endPtr;
					*endPtr = *beginPtr;
					*beginPtr = tmp;
					beginPtr++;
					endPtr--;
				}
				short outVal;
				if (oriVal<-oriMax) {
					outVal = -outMax;
				} else if (oriVal>oriMax) {
					outVal = outMax;
				} else {
					outVal = (((double)oriVal) / oriMax) * outMax;
				}

				beginPtr = static_cast<char*>(static_cast<void*>(&outVal));
				endPtr = beginPtr+1;

				tmp = *endPtr;
				*endPtr = *beginPtr;
				*beginPtr = tmp;

				outBuffer[idx] = outVal;
			}

			fseek(fo, hOut + z*N*2, SEEK_SET);
			fwrite(outBuffer.data(), N, 2, fo);

			z++;
			if ( z >= dimZ ) pthread->cont = 0;
			if ( pthread->abort == 1 ) pthread->cont = 0;
		}

		fclose(fr);
		fclose(fo);
	}
};

MyThread_FloatShort_file_convertion::MyThread_FloatShort_file_convertion() : MyThread_file_convertion() {
}

void MyThread_FloatShort_file_convertion::run() {
	if ( src_filenames.size() == 0 || dst_filenames.size() == 0 ) return;
	cont0 = 1;
	idxfile = 0;
	this->idxfilemax = src_filenames.size();

	while(cont0) {
		QString src = src_filenames[idxfile];
		QString dst = dst_filenames[idxfile];
		inri::Xt xt(src.toStdString());
		abort = xt.is_valid() ? 0 : 1;

		if (abort==0) {
			std::size_t hIn = xt.header_size();
			std::size_t hOut;
			std::size_t dimX = xt.nSamples();
			std::size_t dimY = xt.nRecords();
			std::size_t dimZ = xt.nSlices();

			bool replaceOldFile = QFileInfo(dst).exists();
			{
				inri::Xt outCube(dst.toStdString(), dimX, dimY, dimZ, inri::Xt::Unsigned_16);
				hOut = outCube.header_size();
			}
			float sample0 = xt.startSamples();
			float sampleStep = xt.stepSamples();
			float record0 = xt.startRecord();
			float recordStep = xt.stepRecords();
			float slice0 = xt.startSlice();
			float sliceStep = xt.stepSlices();
			float interRecord = xt.interRecords();
			float interSlice = xt.interSlices();


			int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(src);
			struct nf_fmt ifmt;
			struct image*  im = image_(const_cast<char*>(dst.toStdString().c_str()), "s", " ", &ifmt);

			std::vector<char> _buf;
			int nBuf = 4000;
			_buf.resize(nBuf);
			char* buf = _buf.data();

			copyChar(buf, nBuf, std::to_string(sample0));
			int xx = irephline(im, "TDEB=", 0, buf);
			copyChar(buf, nBuf, std::to_string(sampleStep));
			xx = irephline(im, "PASECH=", 1, buf);
			copyChar(buf, nBuf, std::to_string(record0));
			xx = irephline(im, "NUMTRDB=", 1, buf);
			copyChar(buf, nBuf, std::to_string(recordStep));
			xx = irephline(im, "PASTR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(slice0));
			xx = irephline(im, "NUMPRDB=", 1, buf);
			copyChar(buf, nBuf, std::to_string(sliceStep));
			xx = irephline(im, "PASPR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(interRecord));
			xx = irephline(im, "INTERTR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(interSlice));
			xx = irephline(im, "INTERPR=", 1, buf);

			if (timeOrDepth ==1) {
				//xx = irephline(im, "TYPE_AXE1=", 1, "3");
				//xx = irephline(im, "TYPE_AXE2=", 1, "4");
				//xx = irephline(im, "TYPE_AXE3=", 1, "2");
				xx = irephline(im, "NATUR_DON=", 1, "Depth");
				xx = irephline(im, "TYPE_AXE1=", 1, "2");
				if (replaceOldFile) {
					xx = irephline(im, "Kind=", 1, "Depth");
				} else {
					xx = iputhline(im, "Kind=", "Depth");
				}
			} else {
				//xx = irephline(im, "TYPE_AXE1=", 1, "2");
				//xx = irephline(im, "TYPE_AXE2=", 1, "3");
				//xx = irephline(im, "TYPE_AXE3=", 1, "4");
				xx = irephline(im, "NATUR_DON=", 1, "Time");
				xx = irephline(im, "TYPE_AXE1=", 1, "1");
				if (replaceOldFile) {
					xx = irephline(im, "Kind=", 1, "Time");
				} else {
					xx = iputhline(im, "Kind=", "Time");
				}
			}
			c_fermnf(im);

			ImageFormats::QSampleType sampleType = Seismic3DAbstractDataset::translateType(xt.type());
			SampleTypeBinder binder(sampleType);
			binder.bind<CopyToShortKernel>(this, src, dst, dimX,
					dimY, dimZ, hIn, hOut);

		}

		idxfile++;
		if ( idxfile >= idxfilemax ) cont0 = 0;
		if ( this->abort == 1 ) cont0 = 0;
	}

	complete = 1;
}

template<typename InputType>
struct CopyCharToShortKernel {
	static void run(MyThread_CharShort_file_convertion* pthread, const QString& src, const QString& dst, std::size_t dimX,
			std::size_t dimY, std::size_t dimZ, std::size_t hIn, std::size_t hOut) {
		FILE* fr = fopen(src.toStdString().c_str(), "r");
		FILE* fo = fopen(dst.toStdString().c_str(), "r+");

		std::size_t N = dimX*dimY;
		std::vector<InputType> inBuffer;
		std::vector<short> outBuffer;
		inBuffer.resize(N);
		outBuffer.resize(N);
		pthread->cpt_max = dimZ-1;
		pthread->cont = 1;
		long z=0;

		// apply convertion
		while (pthread->cont) {
			pthread->cpt = z;

			fseek(fr, hIn + z*N, SEEK_SET);
			fread(inBuffer.data(), N, 1, fr);

			#pragma omp parallel for
			for (std::size_t idx=0; idx<N; idx++) {
				InputType oriVal = inBuffer[idx];
				// no swap to do because this is supposed to be for char and unsigned char
				short outVal = oriVal;

				char* beginPtr = static_cast<char*>(static_cast<void*>(&outVal));
				char* endPtr = beginPtr+1;

				char tmp = *endPtr;
				*endPtr = *beginPtr;
				*beginPtr = tmp;

				outBuffer[idx] = outVal;
			}

			fseek(fo, hOut + z*N*2, SEEK_SET);
			fwrite(outBuffer.data(), N, 2, fo);

			z++;
			if ( z >= dimZ ) pthread->cont = 0;
			if ( pthread->abort == 1 ) pthread->cont = 0;
		}

		fclose(fr);
		fclose(fo);
	}
};

MyThread_CharShort_file_convertion::MyThread_CharShort_file_convertion() {
}

void MyThread_CharShort_file_convertion::run() {
	if ( src_filenames.size() == 0 || dst_filenames.size() == 0 ) return;
	cont0 = 1;
	idxfile = 0;
	this->idxfilemax = src_filenames.size();

	while(cont0) {
		QString src = src_filenames[idxfile];
		QString dst = dst_filenames[idxfile];
		inri::Xt xt(src.toStdString());
		abort = xt.is_valid() ? 0 : 1;

		if (abort==0) {
			std::size_t hIn = xt.header_size();
			std::size_t hOut;
			std::size_t dimX = xt.nSamples();
			std::size_t dimY = xt.nRecords();
			std::size_t dimZ = xt.nSlices();

			bool replaceOldFile = QFileInfo(dst).exists();
			{
				inri::Xt outCube(dst.toStdString(), dimX, dimY, dimZ, inri::Xt::Unsigned_16);
				hOut = outCube.header_size();
			}
			float sample0 = xt.startSamples();
			float sampleStep = xt.stepSamples();
			float record0 = xt.startRecord();
			float recordStep = xt.stepRecords();
			float slice0 = xt.startSlice();
			float sliceStep = xt.stepSlices();
			float interRecord = xt.interRecords();
			float interSlice = xt.interSlices();


			int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(src);
			struct nf_fmt ifmt;
			struct image*  im = image_(const_cast<char*>(dst.toStdString().c_str()), "s", " ", &ifmt);

			std::vector<char> _buf;
			int nBuf = 4000;
			_buf.resize(nBuf);
			char* buf = _buf.data();

			copyChar(buf, nBuf, std::to_string(sample0));
			int xx = irephline(im, "TDEB=", 0, buf);
			copyChar(buf, nBuf, std::to_string(sampleStep));
			xx = irephline(im, "PASECH=", 1, buf);
			copyChar(buf, nBuf, std::to_string(record0));
			xx = irephline(im, "NUMTRDB=", 1, buf);
			copyChar(buf, nBuf, std::to_string(recordStep));
			xx = irephline(im, "PASTR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(slice0));
			xx = irephline(im, "NUMPRDB=", 1, buf);
			copyChar(buf, nBuf, std::to_string(sliceStep));
			xx = irephline(im, "PASPR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(interRecord));
			xx = irephline(im, "INTERTR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(interSlice));
			xx = irephline(im, "INTERPR=", 1, buf);

			if (timeOrDepth ==1) {
				//xx = irephline(im, "TYPE_AXE1=", 1, "3");
				//xx = irephline(im, "TYPE_AXE2=", 1, "4");
				//xx = irephline(im, "TYPE_AXE3=", 1, "2");
				xx = irephline(im, "NATUR_DON=", 1, "Depth");
				xx = irephline(im, "TYPE_AXE1=", 1, "2");
				if (replaceOldFile) {
					xx = irephline(im, "Kind=", 1, "Depth");
				} else {
					xx = iputhline(im, "Kind=", "Depth");
				}
			} else {
				//xx = irephline(im, "TYPE_AXE1=", 1, "2");
				//xx = irephline(im, "TYPE_AXE2=", 1, "3");
				//xx = irephline(im, "TYPE_AXE3=", 1, "4");
				xx = irephline(im, "NATUR_DON=", 1, "Time");
				xx = irephline(im, "TYPE_AXE1=", 1, "1");
				if (replaceOldFile) {
					xx = irephline(im, "Kind=", 1, "Time");
				} else {
					xx = iputhline(im, "Kind=", "Time");
				}
			}
			c_fermnf(im);

			ImageFormats::QSampleType sampleType = Seismic3DAbstractDataset::translateType(xt.type());
			SampleTypeBinder binder(sampleType);
			binder.bind<CopyCharToShortKernel>(this, src, dst, dimX,
					dimY, dimZ, hIn, hOut);

		}

		idxfile++;
		if ( idxfile >= idxfilemax ) cont0 = 0;
		if ( this->abort == 1 ) cont0 = 0;
	}

	complete = 1;
}

template<typename InputType>
struct SignalResampleKernel {
	static void run(MyThread_Resample_file_convertion* pthread, const QString& src, const QString& dst, std::size_t dimXIn,
			std::size_t dimXOut, std::size_t dimY, std::size_t dimZ, std::size_t hIn, std::size_t hOut, double stepSampleIn, double stepSampleOut) {
		FILE* fr = fopen(src.toStdString().c_str(), "r");
		FILE* fo = fopen(dst.toStdString().c_str(), "r+");

		std::size_t NIn = dimXIn*dimY;
		std::size_t NOut = dimXOut*dimY;
		std::vector<InputType> inBuffer;
		std::vector<InputType> outBuffer;
		std::vector<double> inBufferFloat;
		std::vector<double> outBufferFloat;

		inBuffer.resize(dimXIn);
		outBuffer.resize(dimXOut);
		inBufferFloat.resize(dimXIn);
		outBufferFloat.resize(dimXOut);
		pthread->cpt_max = dimZ-1;
		pthread->cont = 1;
		long z=0;

		// apply convertion
		while (pthread->cont) {
			pthread->cpt = z;

			fseek(fr, hIn + z*NIn*sizeof(InputType), SEEK_SET);
			fseek(fo, hOut + z*NOut*sizeof(InputType), SEEK_SET);

			for (long y=0; y<dimY; y++) {
				fread(inBuffer.data(), dimXIn, sizeof(InputType), fr);

				for (std::size_t idx=0; idx<dimXIn; idx++) {
					// swap
					InputType val = inBuffer[idx];
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

					inBufferFloat[idx] = val;
				}

				resampleSpline(stepSampleOut, stepSampleIn, inBufferFloat, outBufferFloat);

				if (outBufferFloat.size()!=dimXOut) {
					qDebug() << "Unexpected lenght";
				}

//			#pragma omp parallel for
//			for (std::size_t idx=0; idx<N; idx++) {
//				InputType oriVal = inBuffer[idx];
//				// no swap to do because this is supposed to be for char and unsigned char
//				short outVal = oriVal;
//
//				char* beginPtr = static_cast<char*>(static_cast<void*>(&outVal));
//				char* endPtr = beginPtr+1;
//
//				char tmp = *endPtr;
//				*endPtr = *beginPtr;
//				*beginPtr = tmp;
//
//				outBuffer[idx] = outVal;
//			}
				for (std::size_t idx=0; idx<dimXOut; idx++) {
					// swap
					double valDouble = outBufferFloat[idx];
					InputType val;
					if (valDouble>=std::numeric_limits<InputType>::max()) {
						val = std::numeric_limits<InputType>::max();
					} else if (valDouble<=std::numeric_limits<InputType>::lowest()) {
						val = std::numeric_limits<InputType>::lowest();
					} else {
						val = valDouble;
					}

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

					outBuffer[idx] = val;
				}

				fwrite(outBuffer.data(), dimXOut, sizeof(InputType), fo);
			}

			z++;
			if ( z >= dimZ ) pthread->cont = 0;
			if ( pthread->abort == 1 ) pthread->cont = 0;
		}

		fclose(fr);
		fclose(fo);
	}
};

MyThread_Resample_file_convertion::MyThread_Resample_file_convertion() {
}

void MyThread_Resample_file_convertion::run() {
	if ( src_filenames.size() == 0 || dst_filenames.size() == 0 ) return;
	cont0 = 1;
	idxfile = 0;
	this->idxfilemax = src_filenames.size();

	while(cont0) {
		QString src = src_filenames[idxfile];
		QString dst = dst_filenames[idxfile];
		inri::Xt xt(src.toStdString());
		abort = xt.is_valid() ? 0 : 1;

		if (abort==0) {
			std::size_t hIn = xt.header_size();
			std::size_t hOut;
			std::size_t dimXIn = xt.nSamples();
			std::size_t dimY = xt.nRecords();
			std::size_t dimZ = xt.nSlices();

			float stepSamplesIn = xt.stepSamples();
			float stepSamplesOut = 0.5;

			float stepSampleFactor = stepSamplesIn / stepSamplesOut;

			std::size_t dimXOut = ((int)((dimXIn - 1) * stepSampleFactor)) + 1;

			bool replaceOldFile = QFileInfo(dst).exists();
			{
				inri::Xt outCube(dst.toStdString(), dimXOut, dimY, dimZ, xt.type());
				hOut = outCube.header_size();
			}
			float sample0 = xt.startSamples();
			//float sampleStep = xt.stepSamples();
			float record0 = xt.startRecord();
			float recordStep = xt.stepRecords();
			float slice0 = xt.startSlice();
			float sliceStep = xt.stepSlices();
			float interRecord = xt.interRecords();
			float interSlice = xt.interSlices();


			int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(src);
			struct nf_fmt ifmt;
			struct image*  im = image_(const_cast<char*>(dst.toStdString().c_str()), "s", " ", &ifmt);

			std::vector<char> _buf;
			int nBuf = 4000;
			_buf.resize(nBuf);
			char* buf = _buf.data();

			copyChar(buf, nBuf, std::to_string(sample0));
			int xx = irephline(im, "TDEB=", 0, buf);
			copyChar(buf, nBuf, std::to_string(stepSamplesOut));
			xx = irephline(im, "PASECH=", 1, buf);
			copyChar(buf, nBuf, std::to_string(record0));
			xx = irephline(im, "NUMTRDB=", 1, buf);
			copyChar(buf, nBuf, std::to_string(recordStep));
			xx = irephline(im, "PASTR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(slice0));
			xx = irephline(im, "NUMPRDB=", 1, buf);
			copyChar(buf, nBuf, std::to_string(sliceStep));
			xx = irephline(im, "PASPR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(interRecord));
			xx = irephline(im, "INTERTR=", 1, buf);
			copyChar(buf, nBuf, std::to_string(interSlice));
			xx = irephline(im, "INTERPR=", 1, buf);

			if (timeOrDepth ==1) {
				//xx = irephline(im, "TYPE_AXE1=", 1, "3");
				//xx = irephline(im, "TYPE_AXE2=", 1, "4");
				//xx = irephline(im, "TYPE_AXE3=", 1, "2");
				xx = irephline(im, "NATUR_DON=", 1, "Depth");
				if (replaceOldFile) {
					xx = irephline(im, "Kind=", 1, "Depth");
				} else {
					xx = iputhline(im, "Kind=", "Depth");
				}
			} else {
				//xx = irephline(im, "TYPE_AXE1=", 1, "2");
				//xx = irephline(im, "TYPE_AXE2=", 1, "3");
				//xx = irephline(im, "TYPE_AXE3=", 1, "4");
				xx = irephline(im, "NATUR_DON=", 1, "Time");
				if (replaceOldFile) {
					xx = irephline(im, "Kind=", 1, "Time");
				} else {
					xx = iputhline(im, "Kind=", "Time");
				}
			}
			c_fermnf(im);

			ImageFormats::QSampleType sampleType = Seismic3DAbstractDataset::translateType(xt.type());
			SampleTypeBinder binder(sampleType);
			binder.bind<SignalResampleKernel>(this, src, dst, dimXIn, dimXOut,
					dimY, dimZ, hIn, hOut, stepSamplesIn, stepSamplesOut);
		}

		idxfile++;
		if ( idxfile >= idxfilemax ) cont0 = 0;
		if ( this->abort == 1 ) cont0 = 0;
	}

	complete = 1;
}

FileConversionXTCWT::FileConversionXTCWT(QWidget *parent)
{
	this->setWindowTitle("File conversion Sismage format (xt) to compressed format (cwt)");
	QVBoxLayout * mainLayout=new QVBoxLayout(this);

	QPushButton *qpb_session_load = new QPushButton("Load session");
	QPushButton *qpb_session_save = new QPushButton("Save session");

	projectmanager = new GeotimeProjectManagerWidget(this);
	projectmanager->removeTabHorizons();
	projectmanager->removeTabCulturals();
	projectmanager->removeTabWells();
	projectmanager->removeTabNeurons();
	projectmanager->removeTabPicks();


	QPushButton *qpb_start0 = new QPushButton(this);
	qpb_start0->setText("start");

	mainLayout->addWidget(qpb_session_load);
	mainLayout->addWidget(projectmanager);
	mainLayout->addWidget(qpb_session_save);
	mainLayout->addWidget(qpb_start0);

	connect(qpb_session_load, SIGNAL(clicked()), this, SLOT(trt_session_load()));
	connect(qpb_session_save, SIGNAL(clicked()), this, SLOT(trt_session_save()));
	connect(qpb_start0, SIGNAL(clicked()), this, SLOT(trt_start0()));
}

FileConversionXTCWT::~FileConversionXTCWT()
{

}



void FileConversionXTCWT::trt_start0()
{
	std::vector<QString> name = projectmanager->get_seismic_names();
	std::vector<QString> path = projectmanager->get_seismic_fullpath_names();
	if ( name.size() == 0 ) return;
	DialogFileConvertor *dlg = new DialogFileConvertor(name, path, this);
	dlg->setModal(true);
	dlg->setGeometry(QRect(0, 0, 700, 500));
	dlg->exec();
}

void FileConversionXTCWT::trt_session_load()
{
    projectmanager->load_session_gui();
}


void FileConversionXTCWT::trt_session_save()
{
    projectmanager->save_session_gui();
}




static int getIndexFromVectorString(std::vector<QString> list, QString txt)
{
    for (int i=0; i<list.size(); i++)
    {
        if ( list[i].compare(txt) == 0 )
            return i;
    }
    return -1;
}


// ==============================================
static int file_exists(QString filename)
{
    FILE *file;
	file = fopen(filename.toStdString().c_str(), "r");
    if ( file != NULL )
    {
        fclose(file);
        return 1;
    }
    return 0;
}

static QString dst_filename_create(QString src_filename)
{
	int lastPoint = src_filename.lastIndexOf(".");
	QString fileNameNoExt = src_filename.left(lastPoint);
	QString src_ext = src_filename.right(src_filename.size()-lastPoint-1);
	fprintf(stderr, "ext: %s\n", src_ext.toStdString().c_str());
	QString dst_ext = ".cwt"; if ( src_ext.compare("cwt") == 0 ) dst_ext = ".xt";
	QString dst_filename = fileNameNoExt + dst_ext;
	if ( file_exists(dst_filename) == 0 ) return dst_filename;
	int cpt = 0, cont = 1;
	while ( cont )
	{
		dst_filename = fileNameNoExt + QString("_") + QString::number(cpt++) + dst_ext;
		if ( file_exists(dst_filename) == 0 ) cont = 0;
	}
	return dst_filename;
}


void file_convertion_xt_cwt(QString src_filename, float cwt_error)
{
	/*
	QString dst_filename = dst_filename_create(src_filename);
	fprintf(stderr, "%s -> %s\n", src_filename.toStdString().c_str(), dst_filename.toStdString().c_str());
	FileConversionXTCWT *p = new FileConversionXTCWT(src_filename, dst_filename, cwt_error);
	p->setModal(true);
    p->setGeometry(QRect(0, 0, 260, 190));
    p->exec();		
	*/
}

DialogFileConvertor::DialogFileConvertor(std::vector<QString> name, std::vector<QString> path, QWidget *parent) : QDialog(parent) {
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);
	QTabWidget* tabWidget = new QTabWidget;
	mainLayout->addWidget(tabWidget);

	tabWidget->addTab(new DialogConvertionXTCWT(name, path), "Xt <-> CWT");
	tabWidget->addTab(new DialogFloatToShort(name, path), "Float to Short");
	tabWidget->addTab(new DialogCharToShort(name, path), "Char to Short");
	tabWidget->addTab(new DialogResample(name, path), "Resample");

}

AbstractDialogConverter::AbstractDialogConverter(QWidget* parent) : QWidget(parent) {
	pthread = NULL;
}

void AbstractDialogConverter::buttons_config(bool run)
{
	if ( run )
	{
		qpb_start->setEnabled(false);
		qpb_abort->setEnabled(true);
		qpb_exit->setEnabled(false);
	}
	else
	{
		qpb_start->setEnabled(true);
		qpb_abort->setEnabled(false);
		qpb_exit->setEnabled(true);
	}
}

void AbstractDialogConverter::showTime()
{
	if ( pthread == NULL ) return;

	if ( pthread->idxfile < pthread->idxfilemax )
	{
		QString msg = QString("File: ") + QString::number(pthread->idxfile+1) + " / " + QString::number(pthread->idxfilemax) + QString( " [ ") + pthread->current_filename + QString(" ]");
		this->label_progress->setText(msg);
	}
	else
	{
		this->label_progress->setText("ok");
	}

	float val_f = 100.0*pthread->cpt/pthread->cpt_max;
    int val = (int)(val_f);
    qpb_progress->setValue(val);

	if ( pthread->complete == 1 )
	{
		pthread->complete = 0;
		QMessageBox msgBox;
		msgBox.setText("Info");
		msgBox.setInformativeText("Conversion completed");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.exec();
	}

}

void AbstractDialogConverter::trt_abort()
{
	if ( pthread == NULL ) return;
	QMessageBox msgBox;
	msgBox.setText("Warning");
	msgBox.setInformativeText("Do you really want to abort the process ?");
	msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
	int ret = msgBox.exec();
	if ( ret == QMessageBox::Yes )
	{
		pthread->abort = 1;
	}
	b_run = false;
	buttons_config(b_run);
}

void AbstractDialogConverter::trt_exit()
{
	if ( b_run ) return;
	QWidget::close();
}

DialogConvertionXTCWT::DialogConvertionXTCWT(std::vector<QString> name, std::vector<QString> path, QWidget *parent) : AbstractDialogConverter(parent)
{
	this->pthread = NULL;
	this->b_run = false;
	m_list = name;
	m_path = path;
	QVBoxLayout* mainLayout = new QVBoxLayout(this);

	QLabel *lab = new QLabel("example");
	QHBoxLayout *hh = new QHBoxLayout();
	// hh->addWidget(lab);

	int N = name.size();
	tableWidget = new QTableWidget(N, 2, this);
	// tableWidget->horizontalHeaderItem(0)->setText("File");
	// tableWidget->horizontalHeaderItem(1)->setText("compress factor");
	tableWidget->setHorizontalHeaderItem(0, new QTableWidgetItem("Name"));
	tableWidget->setHorizontalHeaderItem(1, new QTableWidgetItem("factor"));

	QString val;
	for (int n=0; n<N; n++)
	{
		tableWidget->setItem(n, 0, new QTableWidgetItem(name[n]));
		if ( name[n].contains("rgt") )
		{
			val = QString("0.001");
		}
		else
		{
			val = QString("0.01");
		}
		tableWidget->setItem(n, 1, new QTableWidgetItem(val));
	}
	tableWidget->setColumnWidth(0, 300);
	tableWidget->setColumnWidth(1, 200);

	qpb_progress = new QProgressBar;
	qpb_progress->setMinimum(0);
	qpb_progress->setMaximum(100);

	label_progress = new QLabel("progress");

	QHBoxLayout *qhb_buttons = new QHBoxLayout(this);
	qpb_start = new QPushButton("Start process");
	qpb_abort = new QPushButton("Abort process");
	qpb_exit = new QPushButton("Exit");
	qhb_buttons->addWidget(qpb_start);
	qhb_buttons->addWidget(qpb_abort);
	qhb_buttons->addWidget(qpb_exit);

	// hh->addWidget(tableWidget);
	mainLayout->addWidget(tableWidget);
	mainLayout->addWidget(label_progress);
	mainLayout->addWidget(qpb_progress);
	mainLayout->addLayout(qhb_buttons);

	connect(qpb_start, SIGNAL(clicked()), this, SLOT(trt_start()));
	connect(qpb_abort, SIGNAL(clicked()), this, SLOT(trt_abort()));
	connect(qpb_exit, SIGNAL(clicked()), this, SLOT(trt_exit()));

	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	buttons_config(b_run);
}

void DialogConvertionXTCWT::trt_start()
{
	int nbre = m_path.size();
	std::vector<QString> dst_names;
	std::vector<float> err;
	dst_names.resize(nbre);
	err.resize(nbre);
	for (int n=0; n<nbre; n++)
	{
		QString src = m_path[n];
		int lastPoint = src.lastIndexOf(".");
		QString fileNameNoExt = src.left(lastPoint);
		dst_names[n] = fileNameNoExt + QString(".cwt");
		err[n] = tableWidget->item(n, 1)->text().toFloat();
	}
	for (int i=0; i<nbre; i++)
	fprintf(stderr, "%s --> %s [%f]\n", m_path[i].toStdString().c_str(), dst_names[i].toStdString().c_str(), err[i]);
	b_run = true;
	buttons_config(b_run);
	run_conversion(m_path, dst_names, m_list, err);
}

void DialogConvertionXTCWT::run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
std::vector<QString> tiny_filenames, std::vector<float> cwt_error)
{
	MyThread_XTCWT_file_convertion* thread = new MyThread_XTCWT_file_convertion();
	pthread = thread;
	// pthread->src_filename = this->src_filename0;
	// pthread->dst_filename = this->dst_filename0;
	// pthread->cwt_error = this->cwt_error0;
	pthread->src_filenames = src_filenames;
	pthread->dst_filenames = dst_filenames;
	pthread->tiny_filenames = tiny_filenames;
	thread->cwt_error = cwt_error;
    pthread->start();
}

DialogFloatToShort::DialogFloatToShort(std::vector<QString> name, std::vector<QString> path, QWidget *parent) : AbstractDialogConverter(parent)
{
	this->pthread = NULL;
	this->b_run = false;
	//m_list = name;
	//m_path = path;
	int oriN = std::min(name.size(), path.size());
	m_list.resize(oriN);
	m_path.resize(oriN);

	int indexOut = 0;
	for (int indexOri=0; indexOri<oriN; indexOri++) {
		const QString& currentPath = path[indexOri];
		// only allow xt files
		if (QFileInfo(currentPath).suffix().toLower().compare("xt")==0) {
			inri::Xt xt(currentPath.toStdString());
			inri::Xt::Type type = xt.type();
			if (type==inri::Xt::Signed_32 || type==inri::Xt::Unsigned_32 || type==inri::Xt::Signed_64 ||
					type==inri::Xt::Unsigned_64 || type==inri::Xt::Float | type==inri::Xt::Double) {
				m_list[indexOut] = name[indexOri];
				m_path[indexOut] = path[indexOri];
				indexOut++;
			}
		}
	}
	if (indexOut!=oriN) {
		m_list.resize(indexOut);
		m_path.resize(indexOut);
	}

	QVBoxLayout* mainLayout = new QVBoxLayout(this);

	int N = m_list.size();
	tableWidget = new QTableWidget(N, 2, this);
	// tableWidget->horizontalHeaderItem(0)->setText("File");
	// tableWidget->horizontalHeaderItem(1)->setText("compress factor");
	tableWidget->setHorizontalHeaderItem(0, new QTableWidgetItem("Input Name"));
	tableWidget->setHorizontalHeaderItem(1, new QTableWidgetItem("Output Name"));

	QString val;
	for (int n=0; n<N; n++)
	{
		tableWidget->setItem(n, 0, new QTableWidgetItem(m_list[n]));
		tableWidget->setItem(n, 1, new QTableWidgetItem(m_list[n] + "_short"));
	}
	tableWidget->setColumnWidth(0, 300);
	tableWidget->setColumnWidth(1, 200);

	qpb_progress = new QProgressBar;
	qpb_progress->setMinimum(0);
	qpb_progress->setMaximum(100);

	label_progress = new QLabel("progress");

	QHBoxLayout *qhb_buttons = new QHBoxLayout(this);
	qpb_start = new QPushButton("Start process");
	qpb_abort = new QPushButton("Abort process");
	qpb_exit = new QPushButton("Exit");
	qhb_buttons->addWidget(qpb_start);
	qhb_buttons->addWidget(qpb_abort);
	qhb_buttons->addWidget(qpb_exit);

	// hh->addWidget(tableWidget);
	mainLayout->addWidget(tableWidget);
	mainLayout->addWidget(label_progress);
	mainLayout->addWidget(qpb_progress);
	mainLayout->addLayout(qhb_buttons);

	connect(qpb_start, SIGNAL(clicked()), this, SLOT(trt_start()));
	connect(qpb_abort, SIGNAL(clicked()), this, SLOT(trt_abort()));
	connect(qpb_exit, SIGNAL(clicked()), this, SLOT(trt_exit()));

	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	buttons_config(b_run);
}

void DialogFloatToShort::trt_start()
{
	int nbre = m_path.size();
	std::vector<QString> dst_names;
	dst_names.resize(nbre);

	bool valid = true;
	int n=0;
	while (valid && n<nbre)
	{
		QString src = m_path[n];
		QDir dir = QFileInfo(src).dir();
		dst_names[n] = dir.absoluteFilePath("seismic3d." +  tableWidget->item(n, 1)->text() + ".xt");

		if (QFileInfo(dst_names[n]).exists()) {
			QStringList validList;
			validList << tr("Overwrite") << tr("Stop");
			QString item = QInputDialog::getItem(this, tr("File about to be overwritten"), tr("Action:"), validList, 1, false, &valid);
			valid = valid && item.compare(validList[0])==0;
		}
		n++;
	}
	if (valid) {
		for (int i=0; i<nbre; i++)
		fprintf(stderr, "%s --> %s [%f]\n", m_path[i].toStdString().c_str(), dst_names[i].toStdString().c_str());
		b_run = true;
		buttons_config(b_run);

		run_conversion(m_path, dst_names, m_list);
	}
}

void DialogFloatToShort::run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
			std::vector<QString> tiny_filenames) {
	MyThread_FloatShort_file_convertion* thread = new MyThread_FloatShort_file_convertion();

	pthread = thread;
	pthread->src_filenames = src_filenames;
	pthread->dst_filenames = dst_filenames;
	pthread->tiny_filenames = tiny_filenames;
    pthread->start();
}


DialogCharToShort::DialogCharToShort(std::vector<QString> name, std::vector<QString> path, QWidget *parent) : AbstractDialogConverter(parent)
{
	this->pthread = NULL;
	this->b_run = false;
	//m_list = name;
	//m_path = path;
	int oriN = std::min(name.size(), path.size());
	m_list.resize(oriN);
	m_path.resize(oriN);

	int indexOut = 0;
	for (int indexOri=0; indexOri<oriN; indexOri++) {
		const QString& currentPath = path[indexOri];
		// only allow xt files
		if (QFileInfo(currentPath).suffix().toLower().compare("xt")==0) {
			inri::Xt xt(currentPath.toStdString());
			inri::Xt::Type type = xt.type();
			if (type==inri::Xt::Signed_8 || type==inri::Xt::Unsigned_8) {
				m_list[indexOut] = name[indexOri];
				m_path[indexOut] = path[indexOri];
				indexOut++;
			}
		}
	}
	if (indexOut!=oriN) {
		m_list.resize(indexOut);
		m_path.resize(indexOut);
	}

	QVBoxLayout* mainLayout = new QVBoxLayout(this);

	int N = m_list.size();
	tableWidget = new QTableWidget(N, 2, this);
	// tableWidget->horizontalHeaderItem(0)->setText("File");
	// tableWidget->horizontalHeaderItem(1)->setText("compress factor");
	tableWidget->setHorizontalHeaderItem(0, new QTableWidgetItem("Input Name"));
	tableWidget->setHorizontalHeaderItem(1, new QTableWidgetItem("Output Name"));

	QString val;
	for (int n=0; n<N; n++)
	{
		tableWidget->setItem(n, 0, new QTableWidgetItem(m_list[n]));
		tableWidget->setItem(n, 1, new QTableWidgetItem(m_list[n] + "_short"));
	}
	tableWidget->setColumnWidth(0, 300);
	tableWidget->setColumnWidth(1, 200);

	qpb_progress = new QProgressBar;
	qpb_progress->setMinimum(0);
	qpb_progress->setMaximum(100);

	label_progress = new QLabel("progress");

	QHBoxLayout *qhb_buttons = new QHBoxLayout(this);
	qpb_start = new QPushButton("Start process");
	qpb_abort = new QPushButton("Abort process");
	qpb_exit = new QPushButton("Exit");
	qhb_buttons->addWidget(qpb_start);
	qhb_buttons->addWidget(qpb_abort);
	qhb_buttons->addWidget(qpb_exit);

	// hh->addWidget(tableWidget);
	mainLayout->addWidget(tableWidget);
	mainLayout->addWidget(label_progress);
	mainLayout->addWidget(qpb_progress);
	mainLayout->addLayout(qhb_buttons);

	connect(qpb_start, SIGNAL(clicked()), this, SLOT(trt_start()));
	connect(qpb_abort, SIGNAL(clicked()), this, SLOT(trt_abort()));
	connect(qpb_exit, SIGNAL(clicked()), this, SLOT(trt_exit()));

	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	buttons_config(b_run);
}

void DialogCharToShort::trt_start()
{
	int nbre = m_path.size();
	std::vector<QString> dst_names;
	dst_names.resize(nbre);

	bool valid = true;
	int n=0;
	while (valid && n<nbre)
	{
		QString src = m_path[n];
		QDir dir = QFileInfo(src).dir();
		dst_names[n] = dir.absoluteFilePath("seismic3d." +  tableWidget->item(n, 1)->text() + ".xt");

		if (QFileInfo(dst_names[n]).exists()) {
			QStringList validList;
			validList << tr("Overwrite") << tr("Stop");
			QString item = QInputDialog::getItem(this, tr("File about to be overwritten"), tr("Action:"), validList, 1, false, &valid);
			valid = valid && item.compare(validList[0])==0;
		}
		n++;
	}
	if (valid) {
		for (int i=0; i<nbre; i++)
		fprintf(stderr, "%s --> %s [%f]\n", m_path[i].toStdString().c_str(), dst_names[i].toStdString().c_str());
		b_run = true;
		buttons_config(b_run);

		run_conversion(m_path, dst_names, m_list);
	}
}

void DialogCharToShort::run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
			std::vector<QString> tiny_filenames) {
	MyThread_CharShort_file_convertion* thread = new MyThread_CharShort_file_convertion();

	pthread = thread;
	pthread->src_filenames = src_filenames;
	pthread->dst_filenames = dst_filenames;
	pthread->tiny_filenames = tiny_filenames;
    pthread->start();
}

DialogResample::DialogResample(std::vector<QString> name, std::vector<QString> path, QWidget *parent) : AbstractDialogConverter(parent)
{
	this->pthread = NULL;
	this->b_run = false;
	//m_list = name;
	//m_path = path;
	int oriN = std::min(name.size(), path.size());
	m_list.resize(oriN);
	m_path.resize(oriN);

	int indexOut = 0;
	for (int indexOri=0; indexOri<oriN; indexOri++) {
		const QString& currentPath = path[indexOri];
		// only allow xt files
		if (QFileInfo(currentPath).suffix().toLower().compare("xt")==0) {
			inri::Xt xt(currentPath.toStdString());
			inri::Xt::Type type = xt.type();
			m_list[indexOut] = name[indexOri];
			m_path[indexOut] = path[indexOri];
			indexOut++;
		}
	}
	if (indexOut!=oriN) {
		m_list.resize(indexOut);
		m_path.resize(indexOut);
	}

	QVBoxLayout* mainLayout = new QVBoxLayout(this);

	int N = m_list.size();
	tableWidget = new QTableWidget(N, 2, this);
	// tableWidget->horizontalHeaderItem(0)->setText("File");
	// tableWidget->horizontalHeaderItem(1)->setText("compress factor");
	tableWidget->setHorizontalHeaderItem(0, new QTableWidgetItem("Input Name"));
	tableWidget->setHorizontalHeaderItem(1, new QTableWidgetItem("Output Name"));

	QString val;
	for (int n=0; n<N; n++)
	{
		tableWidget->setItem(n, 0, new QTableWidgetItem(m_list[n]));
		tableWidget->setItem(n, 1, new QTableWidgetItem(m_list[n] + "_resample"));
	}
	tableWidget->setColumnWidth(0, 300);
	tableWidget->setColumnWidth(1, 200);

	qpb_progress = new QProgressBar;
	qpb_progress->setMinimum(0);
	qpb_progress->setMaximum(100);

	label_progress = new QLabel("progress");

	QHBoxLayout *qhb_buttons = new QHBoxLayout(this);
	qpb_start = new QPushButton("Start process");
	qpb_abort = new QPushButton("Abort process");
	qpb_exit = new QPushButton("Exit");
	qhb_buttons->addWidget(qpb_start);
	qhb_buttons->addWidget(qpb_abort);
	qhb_buttons->addWidget(qpb_exit);

	// hh->addWidget(tableWidget);
	mainLayout->addWidget(tableWidget);
	mainLayout->addWidget(label_progress);
	mainLayout->addWidget(qpb_progress);
	mainLayout->addLayout(qhb_buttons);

	connect(qpb_start, SIGNAL(clicked()), this, SLOT(trt_start()));
	connect(qpb_abort, SIGNAL(clicked()), this, SLOT(trt_abort()));
	connect(qpb_exit, SIGNAL(clicked()), this, SLOT(trt_exit()));

	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	buttons_config(b_run);
}

void DialogResample::trt_start()
{
	int nbre = m_path.size();
	std::vector<QString> dst_names;
	dst_names.resize(nbre);

	bool valid = true;
	int n=0;
	while (valid && n<nbre)
	{
		QString src = m_path[n];
		QDir dir = QFileInfo(src).dir();
		dst_names[n] = dir.absoluteFilePath("seismic3d." +  tableWidget->item(n, 1)->text() + ".xt");

		if (QFileInfo(dst_names[n]).exists()) {
			QStringList validList;
			validList << tr("Overwrite") << tr("Stop");
			QString item = QInputDialog::getItem(this, tr("File about to be overwritten"), tr("Action:"), validList, 1, false, &valid);
			valid = valid && item.compare(validList[0])==0;
		}
		n++;
	}
	if (valid) {
		for (int i=0; i<nbre; i++)
		fprintf(stderr, "%s --> %s [%f]\n", m_path[i].toStdString().c_str(), dst_names[i].toStdString().c_str());
		b_run = true;
		buttons_config(b_run);

		run_conversion(m_path, dst_names, m_list);
	}
}

void DialogResample::run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
			std::vector<QString> tiny_filenames) {
	MyThread_Resample_file_convertion* thread = new MyThread_Resample_file_convertion();

	pthread = thread;
	pthread->src_filenames = src_filenames;
	pthread->dst_filenames = dst_filenames;
	pthread->tiny_filenames = tiny_filenames;
    pthread->start();
}

