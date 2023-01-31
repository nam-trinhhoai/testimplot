WorkingSetManager::FolderList folders = m_manager->folders();

// Seismic
FolderData* seismics = folders.seismics;
QList<IData*> iData = seismics->data();
SeismicSurvey* seismicSurvey = dynamic_cast<SeismicSurvey*>(iData[i]);
QList<Seismic3DAbstractDataset*> dataset = seismicSurvey->datasets();


FolderData* wells = folders.wells;
QList<IData*> iData = wells->data();
WellBore* bore = wellHead->wellBores()[iWellbore];
Deviations& const deviation = bore->deviations();
double top = deviation.mds[0]; // deviation.mds is a vector
double bottom = deviation.mds.back();

wellbore -> log -> depth

idx = 100
depth[idx];

md, puit, dataset -> seismic_val


m_seismicUnit: unité de cube (temp/profondeur) <- dataset???

m_seismicUnit = dataset[idx]->cubeSeismicAddon().getSampleUnit();

unit = MD; // c'est un enum

m_sampleTransformSurrechantillon = dataset[idx]->sampleTransformation();

m_halfWindow = 0; // can be removed

m_numSamplesSurrechantillon = dataset[idx]->height();

m_ijToXYTransfo = dataset[idx]->ijToXYTransfo();

m_numTraces = dataset[idx]->width();

m_numProfile = dataset[idx]->depth();


int m_halfWindow = 0; // can be removed
int m_numSamplesSurrechantillon = dataset[idx]->height();



Affine2DTransformation* m_ijToXYTransfo = dataset[idx]->ijToXYTransfo();



int m_numTraces = dataset[idx]->width();



int m_numProfile = dataset[idx]->depth();


std::pair<bool, BnniJsonGenerator::IJKPoint> BnniJsonGenerator::isPointInBoundingBox(WellUnit unit, double logKey, WellBore* wellBore) {
	IJKPoint pt;
	bool out = false;

	// get sampleI
	double sampleI;
	sampleI = wellBore->getDepthFromWellUnit(logKey, unit, m_seismicUnit, &out);

	// check i
	if (out) {
		double i;
		m_sampleTransformSurrechantillon->indirect(sampleI, i);
		pt.i = i;

		out = pt.i-m_halfWindow>=0 && pt.i+m_halfWindow<m_numSamplesSurrechantillon;
	}

	// get and check jk
	if (out) {
		double x = wellBore->getXFromWellUnit(logKey, unit, &out);
		double y;
		if (out) {
			y = wellBore->getYFromWellUnit(logKey, unit, &out);
		}
		if (out) {
			double iMap, jMap;
			m_ijToXYTransfo->worldToImage(x, y, iMap, jMap);
			pt.j = iMap;
			pt.k = jMap;
			out = pt.j>=0 && pt.j<m_numTraces && pt.k>=0 && pt.k<m_numProfils;

/**
			if (m_horizonIntervals.size()>0) {
				// search if point is in an interval
				out = false;
				std::list<std::pair<std::vector<float>, std::vector<float>>>::const_iterator intervalIt = m_horizonIntervals.begin();
				std::size_t mapIdx = pt.j + pt.k*m_numTraces;
				while (!out && intervalIt!=m_horizonIntervals.end()) {
					float topVal = intervalIt->first[mapIdx];
					float bottomVal = intervalIt->second[mapIdx];
					out = topVal!=HORIZON_NULL_VALUE && bottomVal!=HORIZON_NULL_VALUE &&
							sampleI >= topVal && sampleI <= bottomVal;
					intervalIt++;
				}
			}
*/


		}
	}

	return std::pair<bool, IJKPoint>(out, pt);
}



pt, out = true: 

/**
CUDAImagePaletteHolder cudai = CUDAImagePaletteHolder(1, dataset[idx]->height(), SampleType::Float32, dataset[idx]->ijToXYTransfo());
QPolygon qp;
qp << QPoint(pt.j, pt.k);

dataset[idx]->loadRandomLine(&cudai, qp);
*/

if(out)// only works for dataset dimV = 1
{
	Seismic3DDataset * seismic3DDataset = dynamic_cast<Seismic3DDataset*>(dataset[idx]);
	float seismic_val; 
	seismic3DDataset.readSubTrace(&seismic_val, pt.i, pt.i+1, pt.j, pt.k, false);
}









