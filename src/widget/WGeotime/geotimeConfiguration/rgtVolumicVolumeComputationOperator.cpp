
#include <malloc.h>
#include <string.h>

#include <Xt.h>
#include <util.h>
#include <dataWeightedInterpolationToFile.hpp>
#include <rgtVolumicVolumeComputationOperator.h>

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




RgtVolumicComputationOperator::RgtVolumicComputationOperator()
{

}

RgtVolumicComputationOperator::RgtVolumicComputationOperator(std::string seismicFilename,
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

void RgtVolumicComputationOperator::setSurveyPath(QString path)
{
	m_surveyPath = path;
	fprintf(stderr, "path: %s\n", path.toStdString().c_str());
}

void RgtVolumicComputationOperator::setData(float *data, int *size)
{
	m_tau = data;
	m_scaleSize = size;
}

void RgtVolumicComputationOperator::setDip(short *dipxy, short *dipxz)
{
	m_dipxy = dipxy;
	m_dipxz = dipxz;
}

void RgtVolumicComputationOperator::setMask2D(char *mask2D)
{
	m_mask2D = mask2D;
}



QString RgtVolumicComputationOperator::name()
{
	return "processing " + QString(m_rgtName.c_str());
}

QString RgtVolumicComputationOperator::getSurveyPath()
{
	return m_surveyPath;
}


// todo
// static on dataResize
void RgtVolumicComputationOperator::dataScale(float* data, char *mask2D_, short *dipxy, short *dipxz, int *size, double vmin, double vmax)
{
	long size0 = (long)size[0]*size[1]*size[2];
	long size02D = (long)size[1]*size[2];
	double mn;
	double mx;
	long addMin = 0, addMax = 0;

	if ( mask2D_ == nullptr )
	{
		mn = data[0];
		mx = data[0];
		for (long add=0; add<size0; add++)
		{
			mx = MAX(mx, data[add]);
			mn = MIN(mn, data[add]);
		}
	}
	else
	{
		int first = 0;
		for (long add=0; add<size02D; add++)
		{
			if ( mask2D_[add] == 0 ) continue;

			for (long x=0; x<size[0]; x++)
			{
				if ( dipxy[add*size[0]+x] == 0 && dipxz[add*size[0]+x] == 0 ) continue;
				if ( first == 0 )
				{
					mn = data[add*size[0]+x];
					mx = data[add*size[0]+x];
					first = 1;
				}
				else
				{
					// mx = MAX(mx, data[add*size[0]+x]);
					// mn = MIN(mn, data[add*size[0]+x]);
					if ( data[add*size[0]+x] < mn )
					{
						mn = data[add*size[0]+x];
						addMin = add*size[0]+x;
					}
					if ( data[add*size[0]+x] > mx )
					{
						mx = data[add*size[0]+x];
						addMax = add*size[0]+x;
					}
				}
			}
		}
	}
	// TODO
	for (long add=0; add<size0; add++)
	{
		double val = (vmax-vmin)/(mx-mn) * ((double)data[add]-mn) + vmin;
		data[add] = (float)val;
	}
	if ( mask2D_ != nullptr )
	{
		for (long add=0; add<size02D; add++)
		{
			if ( mask2D_[add] == 0 )
			{
				for (long x=0; x<size[0]; x++)
				{
					data[add*size[0]+x] = 0;
				}
			}
			else
			{
				for (long x=0; x<size[0]; x++)
				{
					if ( dipxy[add*size[0]+x] == 0 && dipxz[add*size[0]+x] == 0 )
						data[add*size[0]+x] = 0;
				}
			}
		}
	}
}


void RgtVolumicComputationOperator::computeInline(int z, void** buffer)
{
	fprintf(stderr, "rgt inline >> compute: %d\n", z);

	if ( m_tau == nullptr || m_scaleSize == nullptr || m_dipxy == nullptr || m_dipxz == nullptr || m_mask2D == nullptr ) return;

	float *dataIn = (float*)calloc((long)m_scaleSize[0]*m_scaleSize[1]*2, sizeof(float));
	if ( dataIn == nullptr ) return;
	double alphaZ = (double)m_dimz / m_scaleSize[2];
	int z1 = (int)((double)z  / alphaZ);
	z1 = MIN(z1, m_scaleSize[2]-2);
	int z2 = z1 + 1;
	memcpy(dataIn, &m_tau[(long)m_scaleSize[0]*m_scaleSize[1]*z1], m_scaleSize[0]*m_scaleSize[1]*2*sizeof(float));
	for (long add=0; add<(long)m_scaleSize[1]*2; add++)
		for (long x=0; x<m_scaleSize[0]; x++)
			dataIn[add*m_scaleSize[0]+x] += (float)x;
	int sizeIn[3] = { m_scaleSize[0], m_scaleSize[1], 2 };
	int sizeOut[3] = { (int)m_dimx, (int)m_dimy, (int)m_dimz };
	long add3D = (long)z1*m_scaleSize[0]*m_scaleSize[1];
	long add2D = (long)m_scaleSize[1]*z1;
	dataScale(dataIn, &m_mask2D[add2D], &m_dipxy[add3D], &m_dipxz[add3D], sizeIn, 0.0, 32000.0);
	imgWeightedResize<float, short>(dataIn, m_scaleSize, &m_mask2D[add2D], (int*)sizeOut, z%(int)alphaZ, &m_dipxy[add3D], &m_dipxz[add3D], (short*)buffer[0], 1.0);


	short *data = (short*)(buffer[0]);
	for (int y=0; y<m_dimy; y++)
	{
		for (int x=1; x<m_dimx; x++)
		{
			int g = -data[(long)y*m_dimx+x-1] + data[(long)y*m_dimx+x];
			if ( g < 0 )
				data[(long)y*m_dimx+x-1] = 32000;
		}
	}

	if ( dataIn ) free(dataIn);
}

void RgtVolumicComputationOperator::computeXline(int y, void** buffer)
{
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
}

void RgtVolumicComputationOperator::computeRandom(QPolygon traces, void** buffer)
{

}

long RgtVolumicComputationOperator::dimI() { return m_dimx; }

long RgtVolumicComputationOperator::dimJ() { return m_dimy; }

long RgtVolumicComputationOperator::dimK() { return m_dimz; }

long RgtVolumicComputationOperator::dimV() { return 1; }

double RgtVolumicComputationOperator::originI() { return m_startSample; }

double RgtVolumicComputationOperator::originJ() { return m_startRecord; }

double RgtVolumicComputationOperator::originK() { return m_startSlice; }

double RgtVolumicComputationOperator::stepI() { return m_stepSamples; }

double RgtVolumicComputationOperator::stepJ() { return m_stepRecords; }

double RgtVolumicComputationOperator::stepK() { return m_stepSlices; }

ImageFormats::QSampleType RgtVolumicComputationOperator::sampleType() { return ImageFormats::QSampleType::E::UINT16; }

SampleUnit RgtVolumicComputationOperator::sampleUnit() { return m_sampleUnit; }

