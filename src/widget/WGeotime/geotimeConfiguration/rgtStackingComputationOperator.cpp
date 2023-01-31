
#include <malloc.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <Xt.h>
#include <util.h>
#include <surface_stack.h>
#include <iostream>

#include <rgt_utils.h>

#include <dataWeightedInterpolationToFile.hpp>
#include <rgtStackingComputationOperator.h>


static void rawDatasave0(std::string filename, short *data, int *size)
{
	FILE *pf = fopen(filename.c_str(), "w");
	if ( pf == nullptr )
	{
		fprintf(stderr, "filename problem\n");
		return;
	}
	fwrite(data, sizeof(short), (long)size[0]*size[1]*size[2], pf);
	fclose(pf);
	fprintf(stderr, "size: %d %d %d\n", size[0], size[1], size[2]);
}

static void rawDatasave0(std::string filename, short *data, int dimx, int dimy, int dimz)
{
	FILE *pf = fopen(filename.c_str(), "w");
	if ( pf == nullptr )
	{
		fprintf(stderr, "filename problem\n");
		return;
	}
	fwrite(data, sizeof(short), (long)dimx*dimy*dimz, pf);
	fclose(pf);
	fprintf(stderr, "size: %d %d %d\n", dimx, dimy, dimz);
}




RgtStackingComputationOperator::RgtStackingComputationOperator()
{

}

RgtStackingComputationOperator::RgtStackingComputationOperator(std::string seismicFilename,
		std::string rgtFilename,
		std::string rgtName)
{
	m_rgtFilename = seismicFilename;
	m_rgtFilename = rgtFilename;
	m_rgtName = rgtName;
	inri::Xt xt((const char*)seismicFilename.c_str());
	m_dimx = xt.nSamples();
	m_dimy = xt.nRecords();
	m_dimz = xt.nSlices();
	m_startSlice = xt.startSlice();
	m_startRecord = xt.startRecord();
	m_startSample = xt.startSamples();
	m_stepSlices = xt.stepSlices();
	m_stepRecords = xt.stepRecords();
	m_stepSamples = xt.stepSamples();

	inri::Xt::Axis ax = xt.axis();
	if ( ax == inri::Xt::Axis::Time )
		m_sampleUnit = SampleUnit::TIME;
	else
		m_sampleUnit = SampleUnit::DEPTH;
}

void RgtStackingComputationOperator::setSurveyPath(QString path)
{
	m_surveyPath = path;
	fprintf(stderr, "path: %s\n", path.toStdString().c_str());
}

void RgtStackingComputationOperator::setData(float *data, int *size)
{
	m_tau = data;
	m_scaleSize = size;
}

void RgtStackingComputationOperator::setDip(short *dipxy, short *dipxz)
{
	m_dipxy = dipxy;
	m_dipxz = dipxz;
}

void RgtStackingComputationOperator::setMask2D(char *mask2D)
{
	m_mask2D = mask2D;
}



QString RgtStackingComputationOperator::name()
{
	return "processing " + QString(m_rgtName.c_str());
}

QString RgtStackingComputationOperator::getSurveyPath()
{
	return m_surveyPath;
}


// todo
// static on dataResize
void RgtStackingComputationOperator::dataScale(short* data, int *size, double vmin, double vmax)
{
	long size0 = (long)size[0]*size[1]*size[2];
	long size02D = (long)size[1]*size[2];
	double mn;
	double mx;
	long addMin = 0, addMax = 0;

	mn = data[0];
	mx = data[0];
	for (long add=0; add<size0; add++)
	{
		mx = MAX(mx, data[add]);
		mn = MIN(mn, data[add]);
	}

	// TODO
	for (long add=0; add<size0; add++)
	{
		double val = (vmax-vmin)/(mx-mn) * ((double)data[add]-mn) + vmin;
		data[add] = (float)val;
	}
}


void RgtStackingComputationOperator::integrateInplace(short *stack, long dimx, long dimy, long dimz)
{
    if ( stack == nullptr ) return;

    // double *trace = (double*)calloc(height, sizeof(double));
    for (long z=0; z<dimz; z++)
    {
        for (long y=0; y<dimy; y++)
        {
            short *pstack = &stack[dimx*dimy*z+dimx*y];
            double v = pstack[0];
            for (long x=1; x<dimx; x++)
            {
                v += (double)pstack[x];
            }
            if ( v != 0.0 )
            {
                double scale = 32000.0 / v;
                v = pstack[0];
                pstack[0] = (short)(v * scale);
                for (long x=1; x<dimx; x++)
                {
                    v += pstack[x];
                    pstack[x] = (short)(v*scale);
                }
            }
        }
    }
}

void RgtStackingComputationOperator::transposeInplace(short *data, long dimx, long dimy, long dimz)
{
	short *tmp = (short*)calloc(dimx*dimy, sizeof(short));
	if ( tmp == nullptr ) return;
	for (int z=0; z<dimz; z++)
	{
		short *in = &data[dimx*dimy*z];
		for (int y=0; y<dimy; y++)
		{
			for (int x=0; x<dimx; x++)
			{
				tmp[dimx*y+x] = in[dimy*x+y];
			}
		}
		for (int i=0; i<dimx*dimy; i++)
			in[i] = tmp[i];
	}
	free(tmp);
}


void RgtStackingComputationOperator::computeInline(int z, void** buffer)
{
	fprintf(stderr, "rgt inline >> compute: %d\n", z);
	if ( m_rgtStacking == nullptr )
	{
		memset(buffer[0], 0, (long)m_dimx*m_dimy*sizeof(short));
		return;
	}

	int scaleWidth = 0;
	int scaleHeight = 0;
	int scaleDepth = 0;
	int gpuListNbre = 0;
	int *gpuList = nullptr;
	surface_stack_get_stream_size(m_rgtStacking, &scaleWidth, &scaleHeight, &scaleDepth);
	gpuList = surface_stack_get_gpu_list(m_rgtStacking);
	gpuListNbre = surface_stack_get_gpu_list_nbre (m_rgtStacking);
	int decimationFactor = surface_stack_get_decimation_factor(m_rgtStacking);

	int z1 = z / decimationFactor;
	int z2 = MIN(z1+1, scaleDepth-1);

	float *stackF = (float*)calloc((long)2*scaleWidth*scaleHeight, sizeof(float));
	short *stackS = (short*)calloc((long)2*scaleWidth*scaleHeight, sizeof(short));
	short *data2 = (short*)calloc(scaleWidth*scaleHeight, sizeof(short));
	void **d_stack = (void**)calloc(gpuListNbre, sizeof(void*));;

	if ( stackF && stackS &&  data2 && d_stack)
	{
		for (int i=0; i<gpuListNbre; i++)
			d_stack[i] = surface_stack_get_dstack(m_rgtStacking, i);

		for (int i=0; i<gpuListNbre; i++)
		{
			cudaSetDevice(gpuList[i]);
			short *p0 = (short*)(d_stack[i]);
			short *p1 = p0+(long)z1*(long)scaleWidth*scaleHeight;
			cudaError_t cudaErr = cudaMemcpy(data2, p1, (long)scaleWidth*scaleHeight*sizeof(short), cudaMemcpyDeviceToHost);
			for (int i=0; i<scaleWidth*scaleHeight; i++) stackF[i] += (float)data2[i];

			p1 = p0+(long)z2*(long)scaleWidth*scaleHeight;
			cudaErr = cudaMemcpy(data2, p1, (long)scaleWidth*scaleHeight*sizeof(short), cudaMemcpyDeviceToHost);
			for (int i=0; i<scaleWidth*scaleHeight; i++) stackF[scaleWidth*scaleHeight+i] += (float)data2[i];
		}

		for (int add=0; add<scaleWidth*scaleHeight; add++)
			stackS[add] = (short)MIN(stackF[add], 32000.0f);

		integrateInplace(stackS, scaleHeight, scaleWidth, 2);
		double zd = (double)z / decimationFactor;
		int scale_size[] = {scaleWidth, scaleHeight, 1};
		int nativeSize[] = {m_dimy, m_dimx, 1};
		rgt_inline_resize_v2<short>(1, stackS, scale_size, nativeSize, zd-z1, (short*)buffer[0], 1);
	}
	FREE(stackS);
	FREE(data2);
	FREE(d_stack);
	FREE(stackF);
}

void RgtStackingComputationOperator::computeXline(int y, void** buffer)
{
	/*
	fprintf(stderr, "rgt xline >> compute: %d\n", y);
	if ( m_tau == nullptr || m_scaleSize == nullptr || m_dipxy == nullptr || m_dipxz == nullptr || m_mask2D == nullptr ) return;

	float *dataIn = (float*)calloc((long)m_scaleSize[0]*m_scaleSize[2]*2, sizeof(float));
	short *dipxy = (short*)calloc((long)m_scaleSize[0]*m_scaleSize[2]*2, sizeof(short));
	short *dipxz = (short*)calloc((long)m_scaleSize[0]*m_scaleSize[2]*2, sizeof(short));
	char *mask2D = (char*)calloc(m_scaleSize[2]*2, sizeof(char));

	if ( dataIn == nullptr || dipxy == nullptr || dipxz == nullptr || mask2D == nullptr )
	{
		FREE(dataIn)
		FREE(dipxy)
		FREE(dipxz)
		FREE(mask2D)
		return;
	}

	double alphaY = (double)m_dimy / m_scaleSize[1];
	int y1 = (int)((double)y  / alphaY);
	y1 = MIN(y1, m_scaleSize[1]-2);
	int y2 = y1 + 1;

	for (long y=y1; y<=y2; y++)
	{
		for (long z=0; z<m_scaleSize[2]; z++)
		{
			for (long x=0; x<m_scaleSize[0]; x++)
			{
				dataIn[m_scaleSize[0]*m_scaleSize[2]*(y-y1)+m_scaleSize[0]*z+x] = m_tau[m_scaleSize[0]*m_scaleSize[1]*z+m_scaleSize[0]*y+x];
				dipxy[m_scaleSize[0]*m_scaleSize[2]*(y-y1)+m_scaleSize[0]*z+x] = m_dipxz[m_scaleSize[0]*m_scaleSize[1]*z+m_scaleSize[0]*y+x];
				dipxz[m_scaleSize[0]*m_scaleSize[2]*(y-y1)+m_scaleSize[0]*z+x] = m_dipxy[m_scaleSize[0]*m_scaleSize[1]*z+m_scaleSize[0]*y+x];
			}
			mask2D[m_scaleSize[2]*(y-y1)+z] = m_mask2D[m_scaleSize[1]*z+y];
		}
	}

	for (long add=0; add<(long)m_scaleSize[2]*2; add++)
			for (long x=0; x<m_scaleSize[0]; x++)
				dataIn[add*m_scaleSize[0]+x] += (float)x;

	int sizeIn[3] = { m_scaleSize[0], m_scaleSize[2], 2 };
	int sizeOut[3] = { (int)m_dimx, (int)m_dimz, (int)m_dimy };
	dataScale(dataIn, mask2D, dipxy, dipxz, sizeIn, 0.0, 32000.0);
	imgWeightedResize<float, short>(dataIn, sizeIn, mask2D, sizeOut, y%(int)alphaY, dipxy, dipxz, (short*)buffer[0], 1.0);

	FREE(dataIn)
	FREE(dipxy)
	FREE(dipxz)
	FREE(mask2D)
	*/
}

void RgtStackingComputationOperator::computeRandom(QPolygon traces, void** buffer)
{

}

long RgtStackingComputationOperator::dimI() { return m_dimx; }

long RgtStackingComputationOperator::dimJ() { return m_dimy; }

long RgtStackingComputationOperator::dimK() { return m_dimz; }

long RgtStackingComputationOperator::dimV() { return 1; }

double RgtStackingComputationOperator::originI() { return m_startSample; }

double RgtStackingComputationOperator::originJ() { return m_startRecord; }

double RgtStackingComputationOperator::originK() { return m_startSlice; }

double RgtStackingComputationOperator::stepI() { return m_stepSamples; }

double RgtStackingComputationOperator::stepJ() { return m_stepRecords; }

double RgtStackingComputationOperator::stepK() { return m_stepSlices; }

ImageFormats::QSampleType RgtStackingComputationOperator::sampleType() { return ImageFormats::QSampleType::E::UINT16; }

SampleUnit RgtStackingComputationOperator::sampleUnit() { return m_sampleUnit; }

