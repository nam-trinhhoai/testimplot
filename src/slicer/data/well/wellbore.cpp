#include "wellbore.h"
#include "wellboregraphicrepfactory.h"
#include "wellhead.h"
#include "wellpick.h"
#include "algorithm.h"
#include "mtlengthunit.h"

#include <QDir>
#include <QDateTime>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QRegularExpression>

#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>

// list in and out must be of size 2;
void WellBore::getAffineFromList(const double* in, const double* out, double& a, double& b) {
	if (in[0]==in[1]) {
		a = 0;
		b = 0;
	} else {
		a = (out[1] - out[0]) / (in[1] - in[0]);
		b = out[0] - a * in[0];
	}
}

WellBore::WellBore(WorkingSetManager* manager, QString descFile, QString deviationPath,
		std::vector<QString> tfpPaths, std::vector<QString> tfpNames, std::vector<QString> logPaths,
		std::vector<QString> logNames, WellHead* wellHead, QObject* parent) : IData(manager, parent),
		IFileBasedData(descFile) {
	m_wellHead = wellHead;
	m_tfpsFiles = tfpPaths;
	m_tfpsNames = tfpNames;
	m_logsFiles = logPaths;
	m_logsNames = logNames;
	m_descFile = descFile;

	m_stat="";
	m_elev="";
	m_domain="";
	m_ihs="";
	m_datum="";
	m_uwi="";
	m_velocity="";


	GetInfosDescFile(descFile);

	std::pair<QString, double> pair = getNameFromDescFile(descFile);
	m_name = pair.first;
	m_uuid = QUuid::createUuid();





	m_deviations = getDeviationsFromFile(deviationPath).second;
	double dDatum = pair.second - m_deviations.appliedDatum;
	m_deviations.appliedDatum = pair.second;
	bool isIncreasing = true;
	// apply datum
	// convert "tvd datum" to "tvdss"
	for (std::size_t i=0; i<m_deviations.tvds.size(); i++) {
		double precision = 100000.0;
		m_deviations.tvds[i] = std::round((m_deviations.tvds[i] - dDatum)*precision)/precision;
		isIncreasing = isIncreasing && (i==0 || m_deviations.tvds[i]>m_deviations.tvds[i-1]);
	}

	long ND = m_deviations.mds.size();

	m_deviationSplineActive = ND>2;
	m_deviationAffineActive = ND==2;
	if (m_deviationSplineActive) {
		m_acc_deviation = gsl_interp_accel_alloc();
		m_deviation_tvd_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, ND);
		gsl_spline_init(m_deviation_tvd_spline_steffen, m_deviations.mds.data(), m_deviations.tvds.data(), ND);
		m_deviation_x_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, ND);
		gsl_spline_init(m_deviation_x_spline_steffen, m_deviations.mds.data(), m_deviations.xs.data(), ND);
		m_deviation_y_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, ND);
		gsl_spline_init(m_deviation_y_spline_steffen, m_deviations.mds.data(), m_deviations.ys.data(), ND);

		if (isIncreasing) {
			m_acc_deviation_tvd = gsl_interp_accel_alloc();
			m_deviation_tvd2md_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, ND);
			gsl_spline_init(m_deviation_tvd2md_spline_steffen, m_deviations.tvds.data(), m_deviations.mds.data(), ND);
			m_deviationTvd2MdActive = true;
		}
	} else if(m_deviationAffineActive) {
		getAffineFromList(m_deviations.mds.data(), m_deviations.tvds.data(), m_deviation_tvd_a, m_deviation_tvd_b);
		getAffineFromList(m_deviations.mds.data(), m_deviations.xs.data(), m_deviation_x_a, m_deviation_x_b);
		getAffineFromList(m_deviations.mds.data(), m_deviations.ys.data(), m_deviation_y_a, m_deviation_y_b);

		if (m_deviations.tvds[0]<m_deviations.tvds[1]) {
			getAffineFromList(m_deviations.tvds.data(), m_deviations.mds.data(), m_deviation_tvd2md_a, m_deviation_tvd2md_b);
			m_deviationTvd2MdActive = true;
		}
	}

	if(ND>0){
		m_mdFromDeviationBoundMin = m_deviations.mds[0];
		m_mdFromDeviationBoundMax = m_deviations.mds[m_deviations.mds.size()-1];

		if (m_deviationTvd2MdActive) {
			m_tvdFromDeviationBoundMin = m_deviations.tvds[0];
			m_tvdFromDeviationBoundMax = m_deviations.tvds[m_deviations.tvds.size()-1];
		}
	}

	bool isTfpValid = false;
	std::size_t indexTfp = 0;
	while (indexTfp<m_tfpsFiles.size() && !isTfpValid) {
		isTfpValid = selectTFP(indexTfp);
		indexTfp++;
	}
	if (!isTfpValid) {
		//qDebug() << "Well Bore : " << m_name << " has an invalid TFP";
	}

	m_repFactory = new WellBoreGraphicRepFactory(this);
}


WellBore::~WellBore() {
	// tfp free
	if (m_acc_tfp==nullptr) {
		gsl_interp_accel_free(m_acc_tfp);
	}
	if (m_tfp_spline_steffen==nullptr) {
		gsl_spline_free(m_tfp_spline_steffen);
	}
	if (m_acc_tfp_index!=nullptr) {
		gsl_interp_accel_free(m_acc_tfp_index);
	}
	if (m_tfp_index_spline_steffen) {
		gsl_spline_free(m_tfp_index_spline_steffen);
	}

	// deviation free
	if (m_deviation_tvd_spline_steffen!=nullptr) {
		gsl_spline_free(m_deviation_tvd_spline_steffen);
	}
	if (m_deviation_x_spline_steffen!=nullptr) {
		gsl_spline_free(m_deviation_x_spline_steffen);
	}
	if (m_deviation_y_spline_steffen!=nullptr) {
		gsl_spline_free(m_deviation_y_spline_steffen);
	}
	if (m_acc_deviation!=nullptr) {
		gsl_interp_accel_free(m_acc_deviation);
	}
	if (m_acc_deviation_tvd!=nullptr) {
		gsl_interp_accel_free(m_acc_deviation_tvd);
	}
	if (m_deviation_tvd2md_spline_steffen) {
		gsl_spline_free(m_deviation_tvd2md_spline_steffen);
	}

	// log free
	if (m_acc_log!=nullptr) {
		gsl_interp_accel_free(m_acc_log);
	}
	if (m_log_val_spline_steffen!=nullptr) {
		gsl_spline_free(m_log_val_spline_steffen);
	}
}

// Begin MZR 04082021

void WellBore::SetTfpsPath(const std::vector<QString> &tfpPaths)
{
	m_tfpsFiles = tfpPaths;
	bool isTfpValid = false;
	std::size_t indexTfp = 0;
	while (indexTfp<m_tfpsFiles.size() && !isTfpValid) {
		isTfpValid = selectTFP(indexTfp);
		indexTfp++;
	}
	if (!isTfpValid) {
		qDebug() << "Well Bore : " << m_name << " has an invalid TFP";
	}
	emit boreUpdated();
}

void WellBore::SetTfpName(const std::vector<QString> &tfpNames)
{
	m_tfpsNames = tfpNames;
	emit boreUpdated();
}

void WellBore::SetlogPath(const std::vector<QString> &logPaths)
{
	m_logsFiles = logPaths;
	emit boreUpdated();
}

void WellBore::SetlogName(const std::vector<QString> &logNames)
{
	m_logsNames = logNames;
	emit boreUpdated();
}
// End MZR 04082021

IGraphicRepFactory* WellBore::graphicRepFactory() {
	return m_repFactory;
}

QUuid WellBore::dataID() const {
	return m_uuid;
}

const Deviations& WellBore::deviations() const {
	return m_deviations;
}

void WellBore::addPick(WellPick* pick) {
	m_picks.push_back(pick);
	emit pickAdded(pick);
}

void WellBore::removePick(WellPick* pick) {
	m_picks.removeOne(pick);
	emit pickRemoved(pick);
}

QList<WellPick*> WellBore::picks() {
	return m_picks;
}


QVector3D WellBore::getDirectionFromMd(double value, SampleUnit unit, bool* ok)
{
	for(int i=0;i< m_deviations.xs.size()-1;i++)
	{
		if(value >= m_deviations.mds[i] && value <= m_deviations.mds[i+1] )
		{
			bool ok1,ok2;
			float depth1,depth2;
			if(unit == SampleUnit::DEPTH)
			{
				depth1 =  m_deviations.tvds[i];
				depth2 =  m_deviations.tvds[i+1];
			}
			else if(unit == SampleUnit::TIME)
			{
				depth1= getTwtFromMd(m_deviations.mds[i],&ok1);
				depth2= getTwtFromMd(m_deviations.mds[i+1],&ok2);
			}
			else
			{
				ok1 =false;
				ok2=false;
			}


			QVector3D depart( m_deviations.xs[i], depth1,m_deviations.ys[i]);
			QVector3D dest( m_deviations.xs[i+1], depth2,m_deviations.ys[i+1]);
			*ok = ok1 && ok2;
			return (dest-depart);

		}
	}
	*ok= false;
	return QVector3D(0,0,0);

}

double WellBore::getTvdFromMd(double mdVal, bool* ok) {
	double val;
	*ok = mdVal>=m_mdFromDeviationBoundMin && mdVal<=m_mdFromDeviationBoundMax;
	if (*ok) {
		if (m_deviationSplineActive) {
			val = gsl_spline_eval(m_deviation_tvd_spline_steffen, mdVal, m_acc_deviation);
		} else if (m_deviationAffineActive) {
			val = m_deviation_tvd_a * mdVal + m_deviation_tvd_b;
		} else {
			*ok = false;
		}
	}
	return val;
}

double WellBore::getXFromMd(double mdVal, bool* ok) {
	double val;
	*ok = mdVal>=m_mdFromDeviationBoundMin && mdVal<=m_mdFromDeviationBoundMax;
	if (*ok) {
		if (m_deviationSplineActive) {
			val = gsl_spline_eval(m_deviation_x_spline_steffen, mdVal, m_acc_deviation);
		} else if (m_deviationAffineActive) {
			val = m_deviation_x_a * mdVal + m_deviation_x_b;
		} else {
			*ok = false;
		}
	}
	return val;
}

double WellBore::getYFromMd(double mdVal, bool* ok) {
	double val;
	*ok = mdVal>=m_mdFromDeviationBoundMin && mdVal<=m_mdFromDeviationBoundMax;
	if (*ok) {
		if (m_deviationSplineActive) {
			val = gsl_spline_eval(m_deviation_y_spline_steffen, mdVal, m_acc_deviation);
		} else if (m_deviationAffineActive) {
			val = m_deviation_y_a * mdVal + m_deviation_y_b;
		} else {
			*ok = false;
		}
	}
	return val;
}

double WellBore::getMdFromTvd(double tvdVal, bool* ok) {
	double val;
	*ok = m_deviationTvd2MdActive && tvdVal>=m_tvdFromDeviationBoundMin && tvdVal<=m_tvdFromDeviationBoundMax;
	if (*ok) {
		if (m_deviationSplineActive) {
			val = gsl_spline_eval(m_deviation_tvd2md_spline_steffen, tvdVal, m_acc_deviation_tvd);
		} else if (m_deviationAffineActive) {
			val = m_deviation_tvd2md_a * tvdVal + m_deviation_tvd2md_b;
		} else {
			*ok = false;
		}
	}
	return val;
}

double WellBore::getXFromTvd(double tvdVal, bool* ok) {
	double mdVal = getMdFromTvd(tvdVal, ok);
	double val;
	if (*ok) {
		val = getXFromMd(mdVal, ok);
	}
	return val;
}

double WellBore::getYFromTvd(double tvdVal, bool* ok) {
	double mdVal = getMdFromTvd(tvdVal, ok);
	double val;
	if (*ok) {
		val = getYFromMd(mdVal, ok);
	}
	return val;
}

double WellBore::getTwtFromMd(double mdVal, bool* ok) {
	*ok = m_currentTfpIndex!=-1;
	double twtVal;
	if (*ok) {
		if (m_currentTfps.isTvd) {
			double tvdVal = getTvdFromMd(mdVal, ok);
			*ok = *ok && tvdVal>=m_fromTfpBoundMin && tvdVal<=m_fromTfpBoundMax;
			if (*ok) {
				if (m_tfpSplineActive) {
					twtVal = gsl_spline_eval(m_tfp_spline_steffen, tvdVal, m_acc_tfp);
				} else if (m_tfpAffineActive) {
					twtVal = m_tfp_twt_a * tvdVal + m_tfp_twt_b;
				} else {
					*ok = false;
				}
			}
		} else {
			*ok = mdVal>=m_fromTfpBoundMin && mdVal<=m_fromTfpBoundMax;
			if (*ok) {
				if (m_tfpSplineActive) {
					twtVal = gsl_spline_eval(m_tfp_spline_steffen, mdVal, m_acc_tfp);
				} else if (m_tfpAffineActive) {
					twtVal = m_tfp_twt_a * mdVal + m_tfp_twt_b;
				} else {
					*ok = false;
				}
			}
		}
	}
	return twtVal;
}

double WellBore::getTwtFromTvd(double tvdVal, bool* ok) {
	*ok = m_currentTfpIndex!=-1;
	double twtVal;
	if (m_currentTfps.isTvd) {
		*ok = *ok && tvdVal>=m_fromTfpBoundMin && tvdVal<=m_fromTfpBoundMax;
		if (*ok) {
			if (m_tfpSplineActive) {
				twtVal = gsl_spline_eval(m_tfp_spline_steffen, tvdVal, m_acc_tfp);
			} else if (m_tfpAffineActive) {
				twtVal = m_tfp_twt_a * tvdVal + m_tfp_twt_b;
			} else {
				*ok = false;
			}
		}
	} else {
		// tfp index is md
		double mdVal;
		if (*ok) {
			mdVal = getMdFromTvd(tvdVal, ok);
		}
		if (*ok) {
			twtVal = getTwtFromMd(mdVal, ok);
		}
	}
	return twtVal;
}

double WellBore::getTvdFromTwt(double twtVal, bool* ok) {
	*ok = m_currentTfpIndex!=-1;
	double tvdVal;
	if (m_currentTfps.isTvd) {
		*ok = *ok && m_tfpTwt2IndexActive && twtVal>=m_twtFromTfpBoundMin && twtVal<=m_twtFromTfpBoundMax;
		if (*ok) {
			if (m_tfpSplineActive) {
				tvdVal = gsl_spline_eval(m_tfp_index_spline_steffen, twtVal, m_acc_tfp_index);
			} else if (m_tfpAffineActive) {
				tvdVal = m_tfp_twt_index_a * twtVal + m_tfp_twt_index_b;
			} else {
				*ok = false;
			}
		}
	} else {
		// tfp index is md
		double mdVal;
		if (*ok) {
			mdVal = getMdFromTwt(twtVal, ok);
		}
		if (*ok) {
			tvdVal = getTvdFromMd(mdVal, ok);
		}
	}
	return tvdVal;
}

double WellBore::getMdFromTwt(double twtVal, bool* ok) {
	*ok = m_currentTfpIndex!=-1;
	double mdVal;
	if (m_currentTfps.isTvd) {
		double tvdVal;
		if (*ok) {
			tvdVal = getTvdFromTwt(twtVal, ok);
		}
		if (*ok) {
			mdVal = getMdFromTvd(tvdVal, ok);
		}
	} else {
		*ok = *ok && m_tfpTwt2IndexActive && twtVal>=m_twtFromTfpBoundMin && twtVal<=m_twtFromTfpBoundMax;
		if (*ok) {
			if (m_tfpSplineActive) {
				mdVal = gsl_spline_eval(m_tfp_index_spline_steffen, twtVal, m_acc_tfp_index);
			} else if (m_tfpAffineActive) {
				mdVal = m_tfp_twt_index_a * twtVal + m_tfp_twt_index_b;
			} else {
				*ok = false;
			}
		}
	}
	return mdVal;
}

double WellBore::getXFromTwt(double twtVal, bool* ok) {
	double mdVal = getMdFromTwt(twtVal, ok);
	double xVal;
	if (*ok) {
		xVal = getXFromMd(mdVal, ok);
	}
	return xVal;
}

double WellBore::getYFromTwt(double twtVal, bool* ok) {
	double mdVal = getMdFromTwt(twtVal, ok);
	double yVal;
	if (*ok) {
		yVal = getYFromMd(mdVal, ok);
	}
	return yVal;
}

double WellBore::getLogFromMd(double mdVal, bool* ok) {
	*ok = m_currentLogIndex!=-1 && (m_currentLogs.unit==WellUnit::MD || m_currentLogs.unit==WellUnit::TVD ||
			m_currentLogs.unit==WellUnit::TWT);
	double logVal;

	if (*ok) {
		if (m_currentLogs.unit==WellUnit::MD) {
			if (*ok) {
				// search if tvdVal in bounds
				*ok = false;
				std::size_t idx = 0;
				while(!(*ok) && idx<m_currentLogs.nonNullIntervals.size()) {
					*ok = m_currentLogs.keys[m_currentLogs.nonNullIntervals[idx].first]<=mdVal &&
							m_currentLogs.keys[m_currentLogs.nonNullIntervals[idx].second]>=mdVal;
					idx++;
				}
			}

			if (*ok) {
				// only md log index support  filtering for now
				if (m_useFiltering && m_logFilter!=nullptr) {
					logVal = m_logFilter->getFilteredY(mdVal, ok);
				} else {
					logVal = gsl_spline_eval(m_log_val_spline_steffen, mdVal, m_acc_log);
				}
			}
		} else if (m_currentLogs.unit==WellUnit::TVD) {
			double tvdVal = getTvdFromMd(mdVal, ok);
			logVal = getLogFromTvd(tvdVal, ok);
		} else if (m_currentLogs.unit==WellUnit::TWT) {
			double twtVal = getTwtFromMd(mdVal, ok);
			logVal = getLogFromTwt(twtVal, ok);
		} else {
			*ok = false;
		}
	}
	return logVal;
}

double WellBore::getLogFromTvd(double tvdVal, bool* ok)  {
	*ok = m_currentLogIndex!=-1 && (m_currentLogs.unit==WellUnit::TVD || m_currentLogs.unit==WellUnit::MD ||
			m_currentLogs.unit==WellUnit::TWT);
	double logVal;

	if (*ok) {
		if (m_currentLogs.unit==WellUnit::TVD) {
			if (*ok) {
				// search if tvdVal in bounds
				*ok = false;
				std::size_t idx = 0;
				while(!(*ok) && idx<m_currentLogs.nonNullIntervals.size()) {
					*ok = m_currentLogs.keys[m_currentLogs.nonNullIntervals[idx].first]<=tvdVal &&
							m_currentLogs.keys[m_currentLogs.nonNullIntervals[idx].second]>=tvdVal;
					idx++;
				}
			}
			if (*ok) {
				logVal = gsl_spline_eval(m_log_val_spline_steffen, tvdVal, m_acc_log);
			}
		} else if (m_currentLogs.unit==WellUnit::MD) {
			double mdVal = getMdFromTvd(tvdVal, ok);
			if (*ok) {
				logVal = getLogFromMd(mdVal, ok);
			}
		} else if (m_currentLogs.unit==WellUnit::TWT) {
			double twtVal = getTwtFromTvd(tvdVal, ok);
			if (*ok) {
				logVal = getLogFromTwt(twtVal, ok);
			}
		} else {
			*ok = false;
		}
	}
	return logVal;
}

double WellBore::getLogFromTwt(double twtVal, bool* ok) {
	*ok = m_currentLogIndex!=-1 && (m_currentLogs.unit==WellUnit::TWT || m_currentLogs.unit==WellUnit::TVD ||
			m_currentLogs.unit==WellUnit::MD);
	double logVal;
	if (*ok) {
		if (m_currentLogs.unit==WellUnit::TWT) {
			if (*ok) {
				// search if tvdVal in bounds
				*ok = false;
				std::size_t idx = 0;
				while(!(*ok) && idx<m_currentLogs.nonNullIntervals.size()) {
					*ok = m_currentLogs.keys[m_currentLogs.nonNullIntervals[idx].first]<=twtVal &&
							m_currentLogs.keys[m_currentLogs.nonNullIntervals[idx].second]>=twtVal;
					idx++;
				}
			}
			if (*ok) {
				logVal = gsl_spline_eval(m_log_val_spline_steffen, twtVal, m_acc_log);
			}
		} else if (m_currentLogs.unit==WellUnit::MD) {
			double mdVal = getMdFromTwt(twtVal, ok);
			if (*ok) {
				logVal = getLogFromMd(mdVal, ok);
			}
		} else if (m_currentLogs.unit==WellUnit::TVD) {
			double tvdVal = getTvdFromTwt(twtVal, ok);
			if (*ok) {
				logVal = getLogFromTvd(tvdVal, ok);
			}
		} else {
			*ok = false;
		}
	}
	return logVal;
}

double WellBore::getMdFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok) {
	double mdVal;
	if (wellUnit==WellUnit::MD) {
		mdVal = idxVal;
		*ok = true;
	} else if (wellUnit==WellUnit::TVD) {
		mdVal = getMdFromTvd(idxVal, ok);
	} else if (wellUnit==WellUnit::TWT) {
		mdVal = getMdFromTwt(idxVal, ok);
	} else {
		*ok = false;
	}
	return mdVal;
}

double WellBore::getXFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok) {
	double xVal;
	if (wellUnit==WellUnit::MD) {
		xVal = getXFromMd(idxVal, ok);
	} else if (wellUnit==WellUnit::TVD) {
		xVal = getXFromTvd(idxVal, ok);
	} else if (wellUnit==WellUnit::TWT) {
		xVal = getXFromTwt(idxVal, ok);
	} else {
		*ok = false;
	}
	return xVal;
}

double WellBore::getYFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok) {
	double yVal;
	if (wellUnit==WellUnit::MD) {
		yVal = getYFromMd(idxVal, ok);
	} else if (wellUnit==WellUnit::TVD) {
		yVal = getYFromTvd(idxVal, ok);
	} else if (wellUnit==WellUnit::TWT) {
		yVal = getYFromTwt(idxVal, ok);
	} else {
		*ok = false;
	}
	return yVal;
}

double WellBore::getDepthFromWellUnit(double idxVal, WellUnit wellUnit, SampleUnit depthUnit, bool* ok) {
	double depthVal;
	if (depthUnit==SampleUnit::DEPTH) {
		if (wellUnit==WellUnit::MD) {
			depthVal = getTvdFromMd(idxVal, ok);
		} else if (wellUnit==WellUnit::TVD) {
			depthVal = idxVal;
			*ok = true;
		} else if (wellUnit==WellUnit::TWT) {
			depthVal = getTvdFromTwt(idxVal, ok);
		} else {
			*ok = false;
		}
	} else if (depthUnit==SampleUnit::TIME) {
		if (wellUnit==WellUnit::MD) {
			depthVal = getTwtFromMd(idxVal, ok);
		} else if (wellUnit==WellUnit::TVD) {
			depthVal = getTwtFromTvd(idxVal, ok);
		} else if (wellUnit==WellUnit::TWT) {
			depthVal = idxVal;
			*ok = true;
		} else {
			*ok = false;
		}
	} else {
		*ok = false;
	}

	return depthVal;
}

double WellBore::getLogFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok) {
	double logVal;
	if (wellUnit==WellUnit::MD) {
		logVal = getLogFromMd(idxVal, ok);
	} else if (wellUnit==WellUnit::TVD) {
		logVal = getLogFromTvd(idxVal, ok);
	} else if (wellUnit==WellUnit::TWT) {
		logVal = getLogFromTwt(idxVal, ok);
	} else {
		*ok = false;
	}
	return logVal;
}

double WellBore::getWellUnitFromTwt(double twtVal, WellUnit wellUnit, bool* ok) {
	double idxVal;
	if (wellUnit==WellUnit::MD) {
		idxVal = getMdFromTwt(twtVal, ok);
	} else if (wellUnit==WellUnit::TVD) {
		idxVal = getTvdFromTwt(twtVal, ok);
	} else if (wellUnit==WellUnit::TWT) {
		idxVal = twtVal;
	} else {
		*ok = false;
	}
	return idxVal;
}

bool WellBore::isTfpDefined() const {
	return m_currentTfpIndex!=-1;
}


void WellBore::GetInfosDescFile(QString descFile)
{
	QFile file(descFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot read desc file in text format " << descFile;

	}

	QTextStream in(&file);
	while (!in.atEnd()) {
		QString line = in.readLine();
		QStringList lineSplit = line.split("\t");
		if(lineSplit.size()>1 && lineSplit.first().compare("Name")==0) {
			//wellBoreName = lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Datum")==0) {
			m_datum = lineSplit[1];
		}
		else  if(lineSplit.size()>1 && lineSplit.first().compare("Status")==0) {

			m_stat= lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Elev")==0) {

			m_elev= lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("UWI")==0) {
			m_uwi= lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Domain")==0) {
			m_domain= lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Velocity")==0) {
			m_velocity= lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("IHS")==0) {
			m_ihs= lineSplit[1];
		}
	}
}


std::pair<QString, double> WellBore::getNameFromDescFile(QString descFile) {
	QString wellBoreName;
	bool nameFound = false;
	double datum = 0.0;
	bool isDatumFound = false;
	{
		QFile file(descFile);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			qDebug() << "WellBore : cannot read desc file in text format " << descFile;
			return std::pair<QString, double>(QString(""), 0.0);
		}

		QTextStream in(&file);
		while (!in.atEnd() && (!nameFound || !isDatumFound)) {
			QString line = in.readLine();
			QStringList lineSplit = line.split("\t");
			if(lineSplit.size()>1 && lineSplit.first().compare("Name")==0) {
				wellBoreName = lineSplit[1];
				nameFound = true;
			} else if(lineSplit.size()>1 && lineSplit.first().compare("Datum")==0) {
				datum = lineSplit[1].toDouble(&isDatumFound);
				if (!isDatumFound) {
					datum = 0.0; // reset default value if not found
				}
			}

		}
	}
	if (wellBoreName.isNull() || wellBoreName.isEmpty()) {
		qDebug() << "WellBore : unsupported desc file" << descFile;
	}
	return std::pair<QString, double>(wellBoreName, datum);
}

QString WellBore::getTfpFileFromDescFile(QString descFile) {
	QString tfpFilePath = "";
	int tfpNumber = -1;
	bool velocityValid;
	bool velocityFound = false;
	{
		QFile file(descFile);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			qDebug() << "WellBore : cannot read desc file in text format " << descFile;
			return QString("");
		}

		QTextStream in(&file);
		while (!in.atEnd() && (!velocityFound)) {
			QString line = in.readLine();
			QStringList lineSplit = line.split("\t");
			if(lineSplit.size()>1 && lineSplit.first().compare("Velocity")==0) {
				tfpNumber = lineSplit[1].toInt(&velocityValid);
				velocityFound = true;
			}
		}
	}
	if (!velocityFound || !velocityValid) {
		qDebug() << "WellBore : unsupported desc file" << descFile;
	} else {
		QDir wellBoreDir = QFileInfo(descFile).dir();
		QString extension = ".tfp";
		QStringList descFiles = wellBoreDir.entryList(QStringList() << "*"+extension, QDir::Files);
		if (descFiles.size()>0) {
			QString prefix = descFiles[0].split(".").first().split("#").first(); // no "." and "#" in well name
			if (tfpNumber<=0) {
				tfpFilePath = prefix + extension;
			} else {
				tfpFilePath = prefix + "#" + QString::number(tfpNumber) + extension;
			}
			tfpFilePath = wellBoreDir.absoluteFilePath(tfpFilePath);
		}
	}
	return tfpFilePath;
}

std::pair<bool, Deviations> WellBore::getDeviationsFromFile(QString deviationFile) {
	double wellHeadX = m_wellHead->x();
	double wellHeadY = m_wellHead->y();

	bool isMDInFile = false;
	bool isTvdInFile = false;
	bool isDxInFile = false;
	bool isDyInFile = false;
	bool isIncAzInFile = false;
	bool isHeaderFound = false;
	bool deviationValid = false;

	Deviations deviations;
	QFile file(deviationFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot read desc file in text format " << deviationFile;
	} else {
		deviationValid = true;
		std::size_t tvdIdx, mdIdx, dxIdx, dyIdx, incIdx, azIdx, Ncolumn;

		QTextStream in(&file);
		double lastInc = 0.0, lastAz = 0.0;
		bool lastIncSet = false, lastAzSet = false;
		while (!in.atEnd() && deviationValid) {
			QString line = in.readLine();
			QString trimmedLine = line.trimmed();
			if (trimmedLine.isNull() || trimmedLine.isEmpty()) {
				continue;
			}
			QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);

			Deviation deviation;
			int deviationFillCount = 0;
			if (!isHeaderFound /*&& lineSplit.size()>=3 && !isHeaderFound &&
					lineSplit[0].compare("TVD")==0*/) {
				// search for
				std::size_t idx = 0;
				bool tvdFound = false, mdFound = false, dxFound = false, dyFound = false;
				bool incFound = false, azFound = false;
				while (idx<lineSplit.count() && (!tvdFound || !mdFound || !dxFound || !dyFound)) {
					if (!tvdFound && lineSplit[idx].compare("TVD")==0) {
						tvdFound = true;
						tvdIdx = idx;
					} else if (!mdFound && lineSplit[idx].compare("MD")==0) {
						mdFound = true;
						mdIdx = idx;
					} else if (!dxFound && lineSplit[idx].compare("DX")==0) {
						dxFound = true;
						dxIdx = idx;
					} else if (!dyFound && lineSplit[idx].compare("DY")==0) {
						dyFound = true;
						dyIdx = idx;
					} else if (!incFound && lineSplit[idx].compare("INC")==0) {
						incFound = true;
						incIdx = idx;
					} else if (!azFound && lineSplit[idx].compare("AZ")==0) {
						azFound = true;
						azIdx = idx;
					}
					idx++;
				}
				isHeaderFound = (tvdFound && dxFound && dyFound) || (mdFound && incFound && azFound) ||
						(mdFound && dxFound && dyFound);
				if (isHeaderFound) {
					Ncolumn = lineSplit.count();

					isMDInFile = mdFound;
					isTvdInFile = tvdFound;
					isDxInFile = dxFound;
					isDyInFile = dyFound;
					isIncAzInFile = incFound && azFound;
				}
			} else if(isHeaderFound) {
//				for (int index=0; index<lineSplit.size(); index++) {
//					bool ok;
//					double val = lineSplit[index].toDouble(&ok);
//					if (ok && isMDInFile) {
//						if (deviationFillCount==0) {
//							deviation.tvd = val;
//						} else if (deviationFillCount==1) {
//							deviation.md = val;
//						} else if (deviationFillCount==2) {
//							deviation.x = val + wellHeadX;
//						} else if (deviationFillCount==3) {
//							deviation.y = val + wellHeadY;
//						}
//						deviationFillCount++;
//					} else if (ok) {
//						if (deviationFillCount==0) {
//							deviation.tvd = val;
//						} else if (deviationFillCount==1) {
//							deviation.x = val + wellHeadX;
//						} else if (deviationFillCount==2) {
//							deviation.y = val + wellHeadY;
//						}
//						deviationFillCount++;
//					}
//				}
				deviationValid = Ncolumn==lineSplit.size();
				bool mdIncreasing = true;
				float debugMdDiff = -9999.0;
				if (deviationValid) {
					bool ok;
					if (isMDInFile) {
						deviation.md = lineSplit[mdIdx].toDouble(&ok);
						if (deviations.mds.size()>0) {
							debugMdDiff = deviation.md - deviations.mds[deviations.mds.size()-1];
							mdIncreasing = debugMdDiff > 0;
						}
					}
					if (isTvdInFile) {
						deviation.tvd = lineSplit[tvdIdx].toDouble(&ok);
					}
					if (isDxInFile) {
						deviation.x = lineSplit[dxIdx].toDouble(&ok) + wellHeadX;
					}
					if (isDyInFile) {
						deviation.y = lineSplit[dyIdx].toDouble(&ok) + wellHeadY;
					}
					if (isIncAzInFile && isMDInFile) {
						double dMD, I1=0, I2, A1=0, A2, X1=0, Y1=0, Z1=0;
						I2 = lineSplit[incIdx].toDouble(&ok);
						I2 *= M_PI / 180.0;
						if (lastIncSet) {
							I1 = lastInc;
						}
						A2 = lineSplit[azIdx].toDouble(&ok);
						A2 *= M_PI / 180.0;
						if (lastAzSet) {
							A1 = lastAz;
						}
						if (deviations.mds.size()>0) {
							dMD = deviation.md - deviations.mds[deviations.mds.size()-1];
						} else {
							dMD = deviation.md;
						}
						if (deviations.xs.size()>0) {
							X1 = deviations.xs[deviations.xs.size()-1];
						}
						if (deviations.ys.size()>0) {
							Y1 = deviations.ys[deviations.ys.size()-1];
						}
						if (deviations.tvds.size()>0) {
							Z1 = deviations.tvds[deviations.tvds.size()-1];
						}
						double epsilon = 1.0e-30;
						if (std::fabs(I1-I2)<epsilon && std::fabs(I1)<epsilon &&
								std::fabs(A1-A2)<epsilon && std::fabs(A1)<epsilon) {
							if (!isDxInFile) {
								deviation.x = X1;
							}
							if (!isDyInFile) {
								deviation.y = Y1;
							}
							if (!isTvdInFile) {
								deviation.tvd = Z1 + dMD;
							}
						} else {
							double B = std::acos(std::cos(I2 - I1) - (std::sin(I1)*std::sin(I2)*(1-std::cos(A2-A1))));
							double RF = 2 / B * std::tan(B / 2);
							double dX = dMD/2 * (std::sin(I1)*std::sin(A1) + std::sin(I2)*std::sin(A2))*RF;
							double dY = dMD/2 * (std::sin(I1)*std::cos(A1) + std::sin(I2)*std::cos(A2))*RF;
							double dZ = dMD/2 * (std::cos(I1) + std::cos(I2))*RF;
							double X2 = X1 + dX;
							double Y2 = Y1 + dY;
							double Z2 = Z1 + dZ;
							if (!isDxInFile) {
								deviation.x = X2;
							}
							if (!isDyInFile) {
								deviation.y = Y2;
							}
							if (!isTvdInFile) {
								deviation.tvd = Z2;
							}
						}

						lastInc = I2;
						lastIncSet = true;
						lastAz  = A2;
						lastAzSet = true;
					}

					if (mdIncreasing) {
						if (isMDInFile && !isTvdInFile) {
							deviations.mds.push_back(deviation.md);
							deviations.xs.push_back(deviation.x);
							deviations.ys.push_back(deviation.y);
						} else if (/*deviationFillCount==4 &&*/ isMDInFile) {
			//					deviations.push_back(deviation);
							deviations.tvds.push_back(deviation.tvd);
							deviations.mds.push_back(deviation.md);
							deviations.xs.push_back(deviation.x);
							deviations.ys.push_back(deviation.y);
						} else if (/*deviationFillCount==3 &&*/ !isMDInFile) {
							deviations.tvds.push_back(deviation.tvd);
							deviations.xs.push_back(deviation.x);
							deviations.ys.push_back(deviation.y);
						}// else if (deviationFillCount!=0) {
			//				deviationValid = false;
					} else {
						qDebug() << "Ignoring point because md decreased, diff = " << debugMdDiff;
						deviationValid = false;
					}
				}
			}
		}
	}

	if (!isMDInFile) {
		deviations.mds.clear();
		deviations.mds.resize(deviations.tvds.size(), 0);
		if (deviations.tvds.size()>0) {
			deviations.mds[0];
		}
		for (std::size_t i=1; i<deviations.tvds.size(); i++) {
			double dTvd = deviations.tvds[i] - deviations.tvds[i-1];
			double dX = deviations.xs[i] - deviations.xs[i-1];
			double dY = deviations.ys[i] - deviations.ys[i-1];
			deviations.mds[i] = deviations.mds[i-1] +
					std::sqrt(std::pow(dTvd, 2) + std::pow(dX, 2) + std::pow(dY, 2));
		}
	} else if (isMDInFile && isDxInFile && isDyInFile && !isTvdInFile) {
		deviations.tvds.clear();
		deviations.tvds.resize(deviations.mds.size(), 0);
		for (std::size_t i=1; i<deviations.mds.size(); i++) {
			double dMd = deviations.mds[i] - deviations.mds[i-1];
			double dX = deviations.xs[i] - deviations.xs[i-1];
			double dY = deviations.ys[i] - deviations.ys[i-1];
			deviations.tvds[i] = deviations.tvds[i-1] +
					std::sqrt(std::pow(dMd, 2) - std::pow(dX, 2) - std::pow(dY, 2));
		}
	}
	if (!deviationValid || deviations.xs.size()==0) {
		qDebug() << "WellBore : unsupported deviation file " << deviationFile;
	}

	return std::pair<bool, Deviations>(deviationValid, deviations);
}

std::pair<bool, TFPs> WellBore::getTFPFromFile(QString tfpFile) {
	bool tfpValid = false;
	TFPs tfps;

	//QDir dir = QFileInfo(descFile).absoluteDir();

	bool isDataFound = false;
	bool isDatumFound = false;
	bool isVelocityFound = false;

	double datum = 0;
	double velocity = 1;
	double minKey = std::numeric_limits<double>::max();

	QFile file(tfpFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot read tfp file in text format " << tfpFile;
	} else {
		tfpValid = true;

		QTextStream in(&file);
		while (!in.atEnd() && tfpValid) {
			QString line = in.readLine();
			QString trimmedLine = line.trimmed();
			if (trimmedLine.isNull() || trimmedLine.isEmpty()) {
				continue;
			}
			QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);

			if (isDataFound) {
				double tvd, twt;
				int tfpFillCount = 0;
				if(lineSplit.size()>1) {
					for (int index=0; index<lineSplit.size(); index++) {
						bool ok;
						double val = lineSplit[index].toDouble(&ok);
						if (ok) {
							if (tfpFillCount==0) {
								tvd = val;
							} else if (tfpFillCount==1) {
								twt = val;
							}
							tfpFillCount++;
						}
					}
				}
				if (tfpFillCount==2) {
					if (minKey>tvd) {
						minKey = tvd;
					}
//					deviations.push_back(deviation);
					if (tfps.isTvd) {
						tfps.tvds.push_back(tvd);
					} else {
						tfps.mds.push_back(tvd);
					}
					tfps.twts.push_back(twt);
				} else if (tfpFillCount!=0) {
					tfpValid = false;
				}
			} else if (lineSplit.size()==2 && lineSplit[0].compare("Datum")==0) {
				datum = lineSplit[1].toDouble(&isDatumFound);
			} else if (lineSplit.size()==2 && lineSplit[0].compare("ReplacementVelocity")==0) {
				velocity = lineSplit[1].toDouble(&isVelocityFound);
			} else {
				isDataFound = lineSplit.size()==2 && (lineSplit[0].compare("TVD")==0 || lineSplit[0].compare("MD")==0) && lineSplit[1].compare("TWT")==0;
				if (isDataFound) {
					tfps.isTvd = lineSplit[0].compare("TVD")==0;
				}
			}
		}
	}

	tfpValid = tfpValid && isDataFound; // is data not found then tfp is invalid

	// if key is md and there is a negative value, treat it as tvd instead
	if (tfpValid && !tfps.isTvd && minKey<0) {
		tfps.isTvd = true;
		tfps.tvds = tfps.mds;
		tfps.mds.clear();
	}

	if (!tfpValid || tfps.twts.size()==0) {
		qDebug() << "WellBore : unsupported tfp file " << tfpFile;
	} else if (isVelocityFound && qFuzzyIsNull((float) velocity) && !qFuzzyIsNull((float) datum)) {
		qDebug() << "WellBore : unsupported velocity in file " << tfpFile;
	} else if (isDatumFound && isVelocityFound && !qFuzzyIsNull((float) datum)) {
		for (int i=0; i<tfps.twts.size(); i++) {
			double& val = tfps.twts[i];
			val = val - 2* datum / velocity * 1000; // 2* datum / velocity in ms
		}
	}

	return std::pair<bool, TFPs>(tfpValid, tfps);
}

bool WellBore::selectTFP(std::size_t index) {
	if (index<0 || index>=m_tfpsFiles.size()) {
		return false;
	}

	std::pair<bool, TFPs> tfps = getTFPFromFile(m_tfpsFiles[index]);
	bool increasing = true;
	bool twtIncreasing = true;
	int idx = 1;
	if (tfps.second.isTvd) {
		while (increasing && idx<tfps.second.twts.size()) {
			increasing = tfps.second.tvds[idx-1] < tfps.second.tvds[idx];
			twtIncreasing = twtIncreasing && tfps.second.twts[idx-1] < tfps.second.twts[idx];
			if (increasing) {
				idx++;
			}
		}
	} else {
		while (increasing && idx<tfps.second.twts.size()) {
			increasing = tfps.second.mds[idx-1] < tfps.second.mds[idx];
			twtIncreasing = twtIncreasing && tfps.second.twts[idx-1] < tfps.second.twts[idx];
			if (increasing) {
				idx++;
			}
		}
	}
	if (tfps.first && increasing) {
		m_currentTfps = tfps.second;
		m_currentTfpIndex = index;

		// reset gsl objects
		if (m_acc_tfp!=nullptr) {
			gsl_interp_accel_free(m_acc_tfp);
			m_acc_tfp = nullptr;
		}
		if (m_tfp_spline_steffen!=nullptr) {
			gsl_spline_free(m_tfp_spline_steffen);
			m_tfp_spline_steffen = nullptr;
		}
		if (m_acc_tfp_index!=nullptr) {
			gsl_interp_accel_free(m_acc_tfp_index);
			m_acc_tfp_index = nullptr;
		}
		if (m_tfp_index_spline_steffen) {
			gsl_spline_free(m_tfp_index_spline_steffen);
			m_tfp_index_spline_steffen = nullptr;
		}


		std::size_t N = m_currentTfps.twts.size();
		m_tfpSplineActive = N>2;
		m_tfpAffineActive = N==2;
		std::vector<double>* indexTab;
		if (m_currentTfps.isTvd) {
			indexTab = &m_currentTfps.tvds;
		} else {
			indexTab = &m_currentTfps.mds;
		}
		if (m_tfpSplineActive) {
			m_tfp_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, N);
			m_acc_tfp = gsl_interp_accel_alloc();
			gsl_spline_init(m_tfp_spline_steffen, indexTab->data(), m_currentTfps.twts.data(), N);

			if (twtIncreasing) {
				m_tfp_index_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, N);
				m_acc_tfp_index = gsl_interp_accel_alloc();
				gsl_spline_init(m_tfp_index_spline_steffen, m_currentTfps.twts.data(), indexTab->data(), N);
				m_tfpTwt2IndexActive = true;
			}
		} else {
			getAffineFromList(indexTab->data(), m_currentTfps.twts.data(), m_tfp_twt_a, m_tfp_twt_b);

			if (twtIncreasing) {
				m_tfpTwt2IndexActive = true;
				getAffineFromList(m_currentTfps.twts.data(), indexTab->data(), m_tfp_twt_index_a, m_tfp_twt_index_b);
			}
		}

		m_fromTfpBoundMin = (*indexTab)[0];
		m_fromTfpBoundMax = (*indexTab)[indexTab->size()-1];

		if (twtIncreasing) {
			m_twtFromTfpBoundMin = m_currentTfps.twts[0];
			m_twtFromTfpBoundMax = m_currentTfps.twts[m_currentTfps.twts.size()-1];
		}
	}
	return tfps.first && increasing;
}

bool WellBore::selectLog(std::size_t index) {
	if (index<0 || index>=m_logsFiles.size()) {
		if (m_currentLogIndex!=-1) {
			m_currentLogIndex = -1;
			Logs emptyLog;
			emptyLog.unit = WellUnit::MD;
			m_currentLogs = emptyLog;
			emit logChanged();
		}
		return false;
	}
	if (m_currentLogIndex==index) {
		return true;
	}

	std::pair<bool, Logs> logs = getLogsFromFile(m_logsFiles[index]);
	bool increasing = isLogKeyIncreasing(logs.second);
	if (logs.first && increasing) {
		m_currentLogs = logs.second;
		m_currentLogIndex = index;

		// reset gsl objects
		if (m_acc_log!=nullptr) {
			gsl_interp_accel_free(m_acc_log);
		}
		if (m_log_val_spline_steffen!=nullptr) {
			gsl_spline_free(m_log_val_spline_steffen);
		}


		std::vector<double> filteredKeys, filteredAttributes;
		long NL = m_currentLogs.keys.size();
		bool intervalFound = false;
		long start;
		for (long index=0; index<NL; index++) {
			bool isNullValue = m_currentLogs.attributes[index]==m_currentLogs.nullValue ||
					std::isnan(m_currentLogs.attributes[index]);
			if ((intervalFound && isNullValue) || (intervalFound && (index==NL-1))) {
				long end = index-1;
				if (!isNullValue && index==NL-1) {
					// to not reject last point if it is valid
					end = index;
				}
				if (start<=end) {
					m_currentLogs.nonNullIntervals.push_back(std::pair<long, long>(start, end));
				}
				intervalFound = false;
			} else if (!intervalFound && !isNullValue) {
				start = index;
				intervalFound = true;
			}
			if (!isNullValue) {
				filteredKeys.push_back(m_currentLogs.keys[index]);
				filteredAttributes.push_back(m_currentLogs.attributes[index]);
			}
		}
		computeMinMax();

	    m_acc_log = gsl_interp_accel_alloc();
	    m_log_val_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, filteredKeys.size());
	    gsl_spline_init(m_log_val_spline_steffen, filteredKeys.data(), filteredAttributes.data(), filteredKeys.size());
	}
	// filter if needed
	if (m_useFiltering) {
		activateFiltering(m_highcutFrequency);
	}

	emit logChanged();
	return logs.first && increasing;
}


void WellBore::computeMinMax()
{
	//const Logs& log = m_rep->wellBore()->currentLog();
	bool isLogDefined = this->isLogDefined() && m_currentLogs.nonNullIntervals.size()>0;

	double mini=  std::numeric_limits<double>::max();
	double maxi= std::numeric_limits<double>::lowest();


	if(isLogDefined)
	{
		for(int i=0;i< m_currentLogs.nonNullIntervals.size();i++)
		{
			int start = m_currentLogs.nonNullIntervals[i].first;
			int end = m_currentLogs.nonNullIntervals[i].second;
			for(int index=start;index<=end;index+= 1)
			{
				double valeur = m_currentLogs.attributes[index];
				if(valeur <mini ) mini = valeur;
				if(valeur > maxi) maxi = valeur;
			}
		}
	}

	m_mini = mini;
	m_maxi = maxi;

}

long WellBore::currentLogIndex() const {
	return m_currentLogIndex;
}

const Logs& WellBore::currentLog() const {
	return m_currentLogs;
}

bool WellBore::isLogDefined() const {
	return m_currentLogIndex!=-1;
}

std::pair<bool, Logs> WellBore::getLogsFromFile(QString logFile) {
	bool logValid = false;
	bool isDataFound = false;
	bool isUnitFound = false;
	bool isNullValueFound = false;
	Logs logs;

	QFile file(logFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot read log file in text format " << logFile;
	} else {
		logValid = true;

		QTextStream in(&file);
		while (!in.atEnd() && logValid) {
			QString line = in.readLine();
			QString trimmedLine = line.trimmed();
			if (trimmedLine.isNull() || trimmedLine.isEmpty()) {
				continue;
			}
			QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);

			if (isDataFound) {
				double key, attr;
				int logFillCount = 0;
				if(lineSplit.size()>1) {
					for (int index=0; index<lineSplit.size(); index++) {
						bool ok;
						double val = qstringToDouble(lineSplit[index], &ok);
						if (ok) {
							if (logFillCount==0) {
								key = val;
							} else if (logFillCount==1) {
								attr = val;
							}
							logFillCount++;
						}
					}
				}
				if (logFillCount==2) {
//					deviations.push_back(deviation);
					logs.keys.push_back(key);
					logs.attributes.push_back(attr);
				} else if (logFillCount!=0) {
					logValid = false;
				}
			} else if (isUnitFound && isNullValueFound) {
				isDataFound = lineSplit.size()==2 && lineSplit[0].compare("Index")==0 && lineSplit[1].compare("Samples0")==0;
			}  else {
				if (!isUnitFound) {
					isUnitFound = lineSplit.size()==2 && lineSplit[0].compare("Index")==0 && (lineSplit[1].compare("MD")==0 ||
							lineSplit[1].compare("TVD")==0 || lineSplit[1].compare("TWT")==0);
					if (isUnitFound) {
						if (lineSplit[1].compare("MD")==0) {
							logs.unit = WellUnit::MD;
						} else if (lineSplit[1].compare("TVD")==0) {
							logs.unit = WellUnit::TVD;
						} else if (lineSplit[1].compare("TWT")==0) {
							logs.unit = WellUnit::TWT;
						}
					}
				}
				if (!isNullValueFound) {
					isNullValueFound = lineSplit.size()==2 && lineSplit[0].compare("Null")==0;
					if (isNullValueFound) {
						logs.nullValue = qstringToDouble(lineSplit[1], &isNullValueFound);
					}
				}
			}
		}
	}

	logValid = logValid && isDataFound && isUnitFound;

	if (!logValid || logs.keys.size()==0) {
		qDebug() << "WellBore : unsupported log file " << logFile;
	}

	return std::pair<bool, Logs>(logValid, logs);
}

bool WellBore::writeLog(const QString& reflectivityName, const QString& reflectivityKind, const QString& reflectivityPath,
			const Logs& log) {
	if (log.keys.size()==0 || log.attributes.size()==0 || log.attributes.size()!=log.keys.size() || log.unit==WellUnit::UNDEFINED_UNIT) {
		return false;
	}

	QFile file(reflectivityPath);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot write log file in text format " << reflectivityPath;
	} else {
		QTextStream in(&file);

		// AS : hard coded, do not know how to manage it.
		// Should we scan files / detect overwrite and react to it ?
		int versionNumber = 1;
		QDateTime dateTime = QDateTime::currentDateTime();

		in << "/Attributes\n";
		in << "Date\t" << dateTime.toString("MMMM dd, yyyy hh:mm:ss AP ") + dateTime.timeZoneAbbreviation() << "\n";
		in << "Name\t" << reflectivityName << "\n";
		in << "Kind\t" << reflectivityKind << "\n";
		in << "Unit\t--\n";
		in << "Version\t" << QString::number(versionNumber) << "\n";
		in << "Index\t";
		if (log.unit==WellUnit::TWT) {
			in << "TWT";
		} else if (log.unit==WellUnit::TVD) {
			in << "TVD";
		} else {
			// use md as default
			in << "MD";
		}
		in << "\n";
		// nullValue seem like in g format
		in << "Null\t" << doubleToQString(log.nullValue) << "\n";
		in << "\\Attributes\n";
		in << "\n";
		in << "Index\tSamples0\n";

		for (long i=0; i<log.keys.size(); i++) {
			// data is written in exponential format
			QString keyStr = doubleToQString(log.keys[i], true);
			QString attributeStr = doubleToQString(log.attributes[i], true);

			QString keySpacing;
			for (int i=0; i<16-keyStr.size(); i++) {
				keySpacing = keySpacing + " ";
			}
			QString attributeSpacing;
			for (int i=0; i<16-attributeStr.size(); i++) {
				attributeSpacing = attributeSpacing + " ";
			}

			// spacing to reach a size of 16 char with key + key + tab + spacing to reach a size of 16 char with attribute + attibute
			in << keySpacing << keyStr << "\t" << attributeSpacing << attributeStr << "\n";
		}
	}

	return true;
}

void WellBore::deactivateFiltering() {
	m_useFiltering = false;
	m_logFilter.reset(nullptr);
}

void WellBore::activateFiltering(double freq) { // high cut bandpass filter

	m_highcutFrequency = freq;
	if (isLogDefined() && m_currentLogs.unit==WellUnit::MD) {
		std::vector<FilteringOperator::DefinitionSet> logDefinitions;
		for (long intervalIndex=0; intervalIndex<m_currentLogs.nonNullIntervals.size(); intervalIndex++) {
			FilteringOperator::DefinitionSet logDefinition;
			logDefinition.firstX = m_currentLogs.keys[m_currentLogs.nonNullIntervals[intervalIndex].first];
			logDefinition.stepX = 0.1524; // in meter
			double lastX = m_currentLogs.keys[m_currentLogs.nonNullIntervals[intervalIndex].second];

			for (double x=logDefinition.firstX; x<=lastX; x += logDefinition.stepX) {
				double val = gsl_spline_eval(m_log_val_spline_steffen, x, m_acc_log);
				logDefinition.arrayY.push_back(val);
			}
			logDefinitions.push_back(logDefinition);
		}
		if (logDefinitions.size()>0) {
			m_logFilter.reset(new FilteringOperator(logDefinitions, m_highcutFrequency/1000.0));
			m_useFiltering = true;
		} else {
			qDebug() << "WellBore : No data interval to filter.";
		}
	} else {
		m_useFiltering = true;
	}
}

/*WellBore* WellBore::getWellBoreFromDesc(QString descFile, QString deviationPath, QString tfpPath,
		WellHead* wellHead, WorkingSetManager* manager, QObject* parent) {
	QString wellBoreName;
	bool nameFound = false;
	{
		QFile file(descFile);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			return nullptr;
		}

		QTextStream in(&file);
		while (!in.atEnd() && !nameFound) {
			QString line = in.readLine();
			QStringList lineSplit = line.split("\t");
			if(lineSplit.size()>1 && lineSplit.first().compare("Name")==0) {
				wellBoreName = lineSplit[1];
				nameFound = true;
			}
		}
	}

	bool deviationValid = false;
	Deviations deviations;
	if (nameFound) {
		double wellHeadX = wellHead->x();
		double wellHeadY = wellHead->y();

		//QDir dir = QFileInfo(descFile).absoluteDir();
		QString deviationFile = deviationPath; //dir.absoluteFilePath("deviation");

		QFile file(deviationFile);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			qDebug() << "WellBore : No deviation file";
		} else {
			deviationValid = true;

			QTextStream in(&file);
			while (!in.atEnd() && deviationValid) {
				QString line = in.readLine();
				QStringList lineSplit = line.split(QRegExp("\\s+"), Qt::SkipEmptyParts);

				Deviation deviation;
				int deviationFillCount = 0;
				if(lineSplit.size()>1 && lineSplit.first().compare("TVD")!=0) {
					for (int index=0; index<lineSplit.size(); index++) {
						bool ok;
						double val = lineSplit[index].toDouble(&ok);
						if (ok) {
							if (deviationFillCount==0) {
								deviation.tvd = val;
							} else if (deviationFillCount==1) {
								deviation.md = val;
							} else if (deviationFillCount==2) {
								deviation.x = val + wellHeadX;
							} else if (deviationFillCount==3) {
								deviation.y = val + wellHeadY;
							}
							deviationFillCount++;
						}
					}
				}
				if (deviationFillCount==4) {
//					deviations.push_back(deviation);
					deviations.tvds.push_back(deviation.tvd);
					deviations.mds.push_back(deviation.md);
					deviations.xs.push_back(deviation.x);
					deviations.ys.push_back(deviation.y);
				} else if (deviationFillCount!=0) {
					deviationValid = false;
				}
			}
		}
	}



	bool tfpValid = false;
	bool isTvd = false;
	TFPs tfps;
	if (deviationValid) {
		//QDir dir = QFileInfo(descFile).absoluteDir();

		bool isDataFound = false;

		QFile file(tfpPath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			qDebug() << "WellBore : No deviation file";
		} else {
			tfpValid = true;

			QTextStream in(&file);
			while (!in.atEnd() && tfpValid) {
				QString line = in.readLine();
				QStringList lineSplit = line.split(QRegExp("\\s+"), Qt::SkipEmptyParts);

				if (isDataFound) {
					double tvd, twt;
					int tfpFillCount = 0;
					if(lineSplit.size()>1) {
						for (int index=0; index<lineSplit.size(); index++) {
							bool ok;
							double val = lineSplit[index].toDouble(&ok);
							if (ok) {
								if (tfpFillCount==0) {
									tvd = val;
								} else if (tfpFillCount==1) {
									twt = val;
								}
								tfpFillCount++;
							}
						}
					}
					if (tfpFillCount==2) {
	//					deviations.push_back(deviation);
						tfps.tvds.push_back(tvd);
						tfps.twts.push_back(twt);
					} else if (tfpFillCount!=0) {
						tfpValid = false;
					}
				} else {
					isDataFound = lineSplit.size()==2 && (lineSplit[0].compare("TVD")==0 || lineSplit[0].compare("MD")==0) && lineSplit[1].compare("TWT")==0;
					if (isDataFound) {
						isTvd = lineSplit[0].compare("TVD")==0;
					}
				}
			}
		}

		tfpValid = tfpValid && isDataFound; // is data not found then tfp is invalid
	}

	if (nameFound && deviationValid) {
		return new WellBore(manager, wellBoreName, deviations, tfps, wellHead, parent);
	} else {
		return nullptr;
	}
}*/

QString WellBore::getTfpName() const {
	QString tfpName;
	if (m_currentTfpIndex>=0 && m_currentTfpIndex<m_tfpsNames.size()) {
		tfpName = m_tfpsNames[m_currentTfpIndex];
	}
	return tfpName;
}

QString WellBore::getTfpFilePath() const {
	QString tfpFile;
	if (m_currentTfpIndex>=0 && m_currentTfpIndex<m_tfpsNames.size()) {
		tfpFile = m_tfpsFiles[m_currentTfpIndex];
	}
	return tfpFile;
}

bool WellBore::isWellCompatibleForTime(bool verbose) {
	bool isDeviationCorrect = m_deviations.tvds.size()==m_deviations.xs.size() && m_deviations.tvds.size()==m_deviations.ys.size();

	bool areBoundMatching = false;
	if (m_deviations.mds.size()>2) {
		// By getting twt val for md bound from deviation file, the bound from tfp file are checked internally
		getTwtFromMd(m_deviations.mds[0], &areBoundMatching);
		if (areBoundMatching) {
			getTwtFromMd(m_deviations.mds[m_deviations.mds.size()-1], &areBoundMatching);
		}
	}

	bool isValid = isTfpDefined() && isDeviationCorrect;
	if (!isValid) {
		QStringList debugMsg;
		debugMsg << "WellBore: well bore " << name() << "is not valid please check well.";
		if (!isTfpDefined()) {
			debugMsg << "(tfp)";
		}
		if (!isDeviationCorrect) {
			debugMsg << "(deviation file)";
		}
		if (!areBoundMatching) {
			debugMsg << "(deviation bounds vs tfp bounds)";
		}
		qDebug() << debugMsg;
	}
	return isValid;
}

QColor WellBore::logColor() const {
	return m_logColor;
}

void WellBore::setLogColor(QColor color) {
	if (color!=m_logColor) {
		m_logColor = color;
		emit logColorChanged(m_logColor);
	}
}

const std::vector<QString>& WellBore::logsNames() const {
	return m_logsNames;
}

const std::vector<QString>& WellBore::logsFiles() const {
	return m_logsFiles;
}

// MZR 05082021
const std::vector<QString>& WellBore::tfpsNames() const {
	return m_tfpsNames;
}

const std::vector<QString>& WellBore::tfpsPaths() const {
	return m_tfpsFiles;
}

void WellBore::deleteRep(){
    emit deletedMenu();
}

double WellBore::qstringToDouble(const QString& str, bool* ok) {
	double out;
	out = str.toDouble(ok);
	if (!*ok) {
		if (str.toLower().compare("infinity")==0) {
			if (std::numeric_limits<double>::has_infinity) {
				out = std::numeric_limits<double>::infinity();
			} else {
				out = std::numeric_limits<double>::max();
			}
			*ok = true;
		} else if (str.toLower().compare("-infinity")==0) {
			out = std::numeric_limits<double>::lowest();
			*ok = true;
		}
	}

	return out;
}

QString WellBore::doubleToQString(const double& val, bool useExponential) {
	QString out;
	if ((std::numeric_limits<double>::has_infinity && val==std::numeric_limits<double>::infinity()) ||
			(!std::numeric_limits<double>::has_infinity && val==std::numeric_limits<double>::max())) {
		out = "Infinity";
	} else if (val==std::numeric_limits<double>::lowest()) {
		out = "-Infinity";
	} else if (std::isnan(val)) {
		out = "NaN";
	} else if (useExponential) {
		out = QString::number(val, 'E', 8);
	} else {
		out = QString::number(val);
	}

	return out;
}

std::vector<QString> WellBore::extractLogsKinds() const {
	std::vector<QString> logsKinds;
	logsKinds.resize(m_logsFiles.size());
	for (int i=0; i<m_logsFiles.size(); i++) {
		logsKinds[i] = getKindFromLogFile(m_logsFiles[i]);
	}
	return logsKinds;
}

QString WellBore::getKindFromLogFile(QString logFile) {
	bool kindNotFound = true;
	QString kind = "";
	QFile file(logFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot read log file in text format " << logFile;
	} else {
		QTextStream in(&file);
		while (!in.atEnd() && kindNotFound) {
			QString line = in.readLine();
			QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);

			if (kindNotFound) {
				kindNotFound = !(lineSplit.size()==2 && lineSplit[0].compare("Kind")==0);
				if (!kindNotFound) {
					kind = lineSplit[1];
				}
			}
		}
	}

	if (kindNotFound) {
		kind = "KindNotFound";
		qDebug() << "WellBore::extractLogsKinds Could not find kind for " << logFile;
	}
	return kind;
}

QString WellBore::getDirName() const {
	QFileInfo info(m_descFile);
	QDir dir = info.dir();
	return dir.dirName();
}

QString WellBore::getDescPath() const {
	return m_descFile;
}

QString WellBore::getLogFileName(int logIndex) const {
	return QFileInfo(m_logsFiles[logIndex]).baseName();
}

QString WellBore::getConvertedDatum(const MtLengthUnit* newDepthLengthUnit) {
	bool ok;
	// deviation datum should be a value in meter
	double datumValue = m_datum.toDouble(&ok);

	QString outDatum;
	if (ok) {
		double convertedDatum = MtLengthUnit::convert(MtLengthUnit::METRE, *newDepthLengthUnit, datumValue);
		outDatum = QString::number(convertedDatum);
	} else {
		// could not convert to double, use current datum as fallback output
		outDatum = m_datum;
	}
	return outDatum;
}


bool WellBore::isLogKeyIncreasing(const Logs& log) {
	bool increasing = true;
	int idx = 1;
	while (increasing && idx<log.keys.size()) {
		increasing = log.keys[idx-1] < log.keys[idx];
		if (increasing) {
			idx++;
		}
	}
	return increasing;
}

std::vector<std::pair<double, double>> WellBore::getTwtNonNullInterval(const Logs& log) {
	std::vector<std::pair<double, double>> out;
	bool intervalFound = false;
	double start;
	double last;
	long N = log.keys.size();
	for (long index=0; index<N; index++) {
		bool isNullValue = log.attributes[index]==log.nullValue ||
				std::isnan(log.attributes[index]);

		double twt = 0;
		if (!isNullValue) {
			bool ok;
			twt = getDepthFromWellUnit(log.keys[index], log.unit, SampleUnit::TIME, &ok);
			isNullValue = !ok;
		}

		if ((intervalFound && isNullValue) || (intervalFound && (index==N-1))) {
			double end = last;
			if (!isNullValue && index==N-1) {
				// to not reject last point if it is valid
				end = twt;
			}
			if (start<=end) {
				out.push_back(std::pair<double, double>(start, end));
			}
			intervalFound = false;
		} else if (!intervalFound && !isNullValue) {
			start = twt;
			intervalFound = true;
		}

		last = twt;
	}
	return out;
}

std::vector<std::pair<double, double>> WellBore::intervalsIntersection(const std::vector<std::pair<double, double>>& intervalA,
		const std::vector<std::pair<double, double>>& intervalB) {
	std::vector<std::pair<double, double>> out;

	long bIdx = 0;
	for (long aIdx=0; aIdx<intervalA.size(); aIdx++) {
		bool bWentPastA = false;
		while (bIdx<intervalB.size() && !bWentPastA) {
			while (bIdx<intervalB.size() && intervalB[bIdx].second<=intervalA[aIdx].first) {
				bIdx++;
			}

			if (bIdx<intervalB.size() && intervalB[bIdx].second>intervalA[aIdx].first) {
				double leftBound = std::max(intervalA[aIdx].first, intervalB[bIdx].first);
				double rightBound = std::min(intervalA[aIdx].second, intervalB[bIdx].second);

				if (leftBound<rightBound) {
					out.push_back(std::pair<double, double>(leftBound, rightBound));
					bIdx++;
				} else {
					bWentPastA = true;
				}
			}
		}
	}

	return out;
}

gsl_spline* WellBore::getGslObjectsFromLog(const Logs& log) {
	std::vector<double> filteredKeys, filteredAttributes;
	long NL = log.keys.size();
	for (long index=0; index<NL; index++) {
		bool isNullValue = log.attributes[index]==log.nullValue ||
				std::isnan(log.attributes[index]);

		if (!isNullValue) {
			filteredKeys.push_back(log.keys[index]);
			filteredAttributes.push_back(log.attributes[index]);
		}
	}

	gsl_spline* log_val_spline_steffen = gsl_spline_alloc(gsl_interp_steffen, filteredKeys.size());
    gsl_spline_init(log_val_spline_steffen, filteredKeys.data(), filteredAttributes.data(), filteredKeys.size());

    return log_val_spline_steffen;
}

void WellBore::computeNonNullInterval(Logs& log) {
	long NL = log.keys.size();
	bool intervalFound = false;
	long start;
	for (long index=0; index<NL; index++) {
		bool isNullValue = log.attributes[index]==log.nullValue ||
				std::isnan(log.attributes[index]);
		if ((intervalFound && isNullValue) || (intervalFound && (index==NL-1))) {
			long end = index-1;
			if (!isNullValue && index==NL-1) {
				// to not reject last point if it is valid
				end = index;
			}
			if (start<=end) {
				log.nonNullIntervals.push_back(std::pair<long, long>(start, end));
			}
			intervalFound = false;
		} else if (!intervalFound && !isNullValue) {
			start = index;
			intervalFound = true;
		}
	}
}

// can be made more generic to support an interval of unknown type, here we only tackle the case of a twt interval
// instead of creating another one it is better to improve this function
// this function work well if the delta needed is very small (close to double precision) else it will be extremely slow
// this can be improved by using dichotomy
std::vector<std::pair<double, double>> WellBore::adjustBoundsForTwt(const std::vector<std::pair<double, double>>& intervals,
		WellUnit convertedUnit) {
	std::vector<std::pair<double, double>> modifiedIntervals;
	for (long i=0; i<intervals.size(); i++) {
		const std::pair<double, double>& interval = intervals[i];
		double firstVal = interval.first;
		double lastVal = interval.second;

		bool valid = false;
		while (!valid && firstVal<lastVal) {
			getWellUnitFromTwt(firstVal, convertedUnit, &valid);
			if (!valid) {
				//firstVal += std::numeric_limits<double>::min();
				firstVal = std::nextafter(firstVal, std::numeric_limits<double>::max());
			}
		}

		if (valid) {
			valid = false;
			while (!valid && firstVal<lastVal) {
				getWellUnitFromTwt(lastVal, convertedUnit, &valid);
				if (!valid) {
					//lastVal -= std::numeric_limits<double>::min();
					lastVal = std::nextafter(lastVal, std::numeric_limits<double>::lowest());
				}
			}
		}

		if (valid) {
			modifiedIntervals.push_back(std::pair<double, double>(firstVal, lastVal));
		}
	}

	return modifiedIntervals;
}

ReflectivityError WellBore::computeReflectivity(const QString& rhobPath, const QString& velocityPath, double pasech, double freq,
		bool useRicker, const QString& reflectivityName, const QString& reflectivityKind, const QString& reflectivityPath) {
	double epsilon = 0.01; // use to change a little the key value of logs because "log key -> twt -> log key" is not the identity because of gsl choice

	if (!isTfpDefined()) {
		// computation is done in twt, so twt needs to be defined
		return ReflectivityError::NoTfp;
	}
	if (!m_tfpTwt2IndexActive) {
		return ReflectivityError::TfpNotReversible;
	}

	std::pair<bool, Logs> rhobLog = getLogsFromFile(rhobPath);
	if (!rhobLog.first || !isLogKeyIncreasing(rhobLog.second)) {
		return ReflectivityError::AttributeLogNotValid;
	}

	std::pair<bool, Logs> velocityLog = getLogsFromFile(velocityPath);
	if (!velocityLog.first || !isLogKeyIncreasing(velocityLog.second)) {
		return ReflectivityError::VelocityLogNotValid;
	}

	std::vector<std::pair<double, double>> rhobNonNullIntervals = getTwtNonNullInterval(rhobLog.second);
	std::vector<std::pair<double, double>> velocityNonNullIntervals = getTwtNonNullInterval(velocityLog.second);

	std::vector<std::pair<double, double>> reflectivityNonNullIntervals = intervalsIntersection(rhobNonNullIntervals, velocityNonNullIntervals);

	reflectivityNonNullIntervals = adjustBoundsForTwt(reflectivityNonNullIntervals, rhobLog.second.unit);
	if (velocityLog.second.unit!=rhobLog.second.unit) {
		reflectivityNonNullIntervals = adjustBoundsForTwt(reflectivityNonNullIntervals, velocityLog.second.unit);
	}

	if (reflectivityNonNullIntervals.size()==0) {
		return ReflectivityError::NoLogIntervalIntersection;
	}

	if (rhobLog.second.keys.size()<3) {
		return ReflectivityError::AttributeLogNotValid;
	}

	if (velocityLog.second.keys.size()<3) {
		return ReflectivityError::VelocityLogNotValid;
	}

	computeNonNullInterval(rhobLog.second);
	computeNonNullInterval(velocityLog.second);

	gsl_interp_accel* rhobAcc = gsl_interp_accel_alloc();
	gsl_spline* rhobGsl = getGslObjectsFromLog(rhobLog.second);
	gsl_interp_accel* velocityAcc = gsl_interp_accel_alloc();
	gsl_spline* velocityGsl = getGslObjectsFromLog(velocityLog.second);

	Logs reflectivityLog;
	reflectivityLog.nullValue = std::numeric_limits<double>::quiet_NaN();
	reflectivityLog.unit = WellUnit::TWT;
	for (long intervalIdx = 0; intervalIdx<reflectivityNonNullIntervals.size(); intervalIdx++) {
		long n = std::floor((reflectivityNonNullIntervals[intervalIdx].second - reflectivityNonNullIntervals[intervalIdx].first) / pasech);
		if (n<2) {
			continue;
		}

		std::vector<float> rhob;
		std::vector<float> velocity;
		std::vector<float> reflectivityTab;

		rhob.resize(n);
		velocity.resize(n);
		reflectivityTab.resize(n);

		bool intervalValid = true;
		long i=0;
		while (intervalValid && i<n) {
			double twt = reflectivityNonNullIntervals[intervalIdx].first + pasech * i;
			double rhobIndex = getWellUnitFromTwt(twt, rhobLog.second.unit, &intervalValid);
			bool rhobIndexValid = false;
			int rhobIntervalIdx = 0;
			while (rhobIntervalIdx<rhobLog.second.nonNullIntervals.size() && !rhobIndexValid) {
				std::pair<long, long> limits = rhobLog.second.nonNullIntervals[rhobIntervalIdx];
				rhobIndexValid = rhobIndex>=rhobLog.second.keys[limits.first] &&
						rhobIndex<=rhobLog.second.keys[limits.second];
				if (!rhobIndexValid && rhobIndex>=rhobLog.second.keys[limits.first] && rhobIndex-rhobLog.second.keys[limits.second]<epsilon) {
					rhobIndex = rhobLog.second.keys[limits.second];
					rhobIndexValid = true;
				}
				if (!rhobIndexValid && rhobIndex<=rhobLog.second.keys[limits.second] && rhobLog.second.keys[limits.first]-rhobIndex<epsilon) {
					rhobIndex = rhobLog.second.keys[limits.first];
					rhobIndexValid = true;
				}
				rhobIntervalIdx++;
			}
			double velocityIndex;
			bool velocityIndexValid = false;
			if (intervalValid) {
				velocityIndex = getWellUnitFromTwt(twt, velocityLog.second.unit, &intervalValid);
				int velocityIntervalIdx = 0;
				while (velocityIntervalIdx<velocityLog.second.nonNullIntervals.size() && !velocityIndexValid) {
					std::pair<long, long> limits = velocityLog.second.nonNullIntervals[velocityIntervalIdx];
					velocityIndexValid = velocityIndex>=velocityLog.second.keys[limits.first] &&
							velocityIndex<=velocityLog.second.keys[limits.second];
					if (!velocityIndexValid && velocityIndex>=velocityLog.second.keys[limits.first] && velocityIndex-velocityLog.second.keys[limits.second]<epsilon) {
						velocityIndex = velocityLog.second.keys[limits.second];
						velocityIndexValid = true;
					}
					if (!velocityIndexValid && velocityIndex<=velocityLog.second.keys[limits.second] && velocityLog.second.keys[limits.first]-velocityIndex<epsilon) {
						velocityIndex = velocityLog.second.keys[limits.first];
						velocityIndexValid = true;
					}
					velocityIntervalIdx++;
				}
			}
			intervalValid = velocityIndexValid && rhobIndexValid;

			if (intervalValid) {
				double rhobVal = gsl_spline_eval(rhobGsl, rhobIndex, rhobAcc);
				double velocityVal = gsl_spline_eval(velocityGsl, velocityIndex, velocityAcc);

				rhob[i] = rhobVal;
				velocity[i] = velocityVal;
			}
			i++;
		}

		if (!intervalValid) {
			qDebug() << "WellBore::computeReflectivity : unexpected invalid interval.";
		} else {
			if (useRicker) {
				reflectivityFFTW(velocity.data(), rhob.data(), pasech, freq, reflectivityTab.data(), n);
			} else {
				reflectivity(velocity.data(), rhob.data(), pasech, 0, 1, 0, freq, reflectivityTab.data(), n);
			}

			long offsetOut = reflectivityLog.keys.size();
			bool resized = false;
			if (offsetOut>0) {
				// add nullValue to separate intervals
				double lastIndex = reflectivityLog.keys[offsetOut-1];
				double newFirstIndex = reflectivityNonNullIntervals[intervalIdx].first;

				if (newFirstIndex-lastIndex>0) {
					reflectivityLog.keys.resize(offsetOut+n+1);
					reflectivityLog.attributes.resize(offsetOut+n+1);
					reflectivityLog.keys[offsetOut] = (newFirstIndex-lastIndex)/2 + lastIndex;
					reflectivityLog.attributes[offsetOut] = reflectivityLog.nullValue;
					offsetOut++;
					resized = true;
				}
			}
			if (!resized) {
				reflectivityLog.keys.resize(offsetOut+n);
				reflectivityLog.attributes.resize(offsetOut+n);
			}
			for (long indexOut=0; indexOut<n; indexOut++) {
				reflectivityLog.keys[indexOut+offsetOut] = reflectivityNonNullIntervals[intervalIdx].first + pasech * indexOut;
				reflectivityLog.attributes[indexOut+offsetOut] = reflectivityTab[indexOut];
			}
		}
	}

	gsl_interp_accel_free(rhobAcc);
	gsl_spline_free(rhobGsl);
	gsl_interp_accel_free(velocityAcc);
	gsl_spline_free(velocityGsl);

	if (reflectivityLog.keys.size()==0) {
		return ReflectivityError::OnlyInvalidIntervals;
	}

	bool writeOut = writeLog(reflectivityName, reflectivityKind, reflectivityPath, reflectivityLog);
	if (!writeOut) {
		return ReflectivityError::FailToWriteLog;
	}
	return ReflectivityError::NoError;
}

FilteringOperator::FilteringOperator(const std::vector<DefinitionSet>& intervals, double freq) {
	std::vector<double> xs, ys;

	long N = 0;
	for (int intervalIndex = 0; intervalIndex<intervals.size(); intervalIndex++) {
		N += intervals[intervalIndex].arrayY.size();
	}

	xs.resize(N);
	ys.resize(N);

	m_limits.resize(intervals.size());
	long offset = 0;
	for (int intervalIndex = 0; intervalIndex<intervals.size(); intervalIndex++) {
		// remove zero padding to improve filtering for logs not starting and ending with zeros
		std::vector<double> buffer;
		buffer.resize(intervals[intervalIndex].arrayY.size()/*+2*/);
//		buffer[0] = 0;
//		buffer[1] = 0;
		memcpy(buffer.data()/*+2*/, intervals[intervalIndex].arrayY.data(),
				intervals[intervalIndex].arrayY.size() * sizeof(double));

		double pixelFreq = freq * intervals[intervalIndex].stepX;
		buffer = highcut(pixelFreq, 1, 0, buffer);

		memcpy(ys.data()+offset, buffer.data()/*+2*/, intervals[intervalIndex].arrayY.size() * sizeof(double));
		for (long index=0; index<intervals[intervalIndex].arrayY.size(); index++) {
			xs[index+offset] = intervals[intervalIndex].firstX + index * intervals[intervalIndex].stepX;
		}
		m_limits[intervalIndex] = std::pair<double, double>(intervals[intervalIndex].firstX,
				intervals[intervalIndex].firstX +
				(intervals[intervalIndex].arrayY.size()-1) * intervals[intervalIndex].stepX);
		offset += intervals[intervalIndex].arrayY.size();
	}

    m_acc = gsl_interp_accel_alloc();
    m_spline = gsl_spline_alloc(gsl_interp_steffen, N);
    gsl_spline_init(m_spline, xs.data(), ys.data(), N);
}

FilteringOperator::~FilteringOperator() {
	if (m_acc!=nullptr) {
		gsl_interp_accel_free(m_acc);
	}
	if (m_spline!=nullptr) {
		gsl_spline_free(m_spline);
	}
}

double FilteringOperator::getFilteredY(double x, bool* ok) const {
	*ok = false;
	double out;
	int i=0;
	while (!*ok && i<m_limits.size()) {
		*ok= x>=m_limits[i].first && m_limits[i].second>=x;
		if (*ok) {
			out = gsl_spline_eval(m_spline, x, m_acc);
		}
		i++;
	}
	return out;
}
bool FilteringOperator::isDefined(double x) const {
	bool out = false;
	int i=0;
	while (!out && i<m_limits.size()) {
		out = x>=m_limits[i].first && m_limits[i].second>=x;
		i++;
	}
	return out;
}
