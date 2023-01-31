#include "wellviewer.h"
#include "ui_wellviewer.h"
#include "MuratCanvas2dLogView.h"
#include "Canvas2dVerticalSync.h"
#include "logstats.h"
#include <algorithm>


#include <QGridLayout>
#include <QVBoxLayout>
#include <QLayout>
#include <QImage>
#include <QDebug>
#include <QSplitter>

#include <cmath>
#include <limits>

WellViewer::WellViewer(QWidget *parent) :
    QWidget(parent), synchroniser(this),
    ui(new Ui::WellViewer)
{
    ui->setupUi(this);
    connect(ui->wells, &QListWidget::itemChanged, this, &WellViewer::wellChange);
    connect(ui->wells, &QListWidget::itemSelectionChanged, this, &WellViewer::wellSelectedChanged);

    viewLayout = new QGridLayout;
    holder = new QVBoxLayout;
    ui->holder->setLayout(holder);
    holder->addLayout(viewLayout);

    connect(ui->addPushButton, SIGNAL(clicked()), this, SLOT(addNewRange()));
    connect(ui->removePushButton, SIGNAL(clicked()), this, SLOT(removeRange()));
}

WellViewer::~WellViewer()
{
    delete ui;
}

void WellViewer::updateWells(QVector<Well >* wells) {
    reset();
    this->wells = wells;
    ui->wells->clear();

    for (int i=0; i<wells->size(); i++) {

        QString name = (*wells)[i].name.split("\t").last().split("||").last();
        QListWidgetItem* item = new QListWidgetItem(name);
        item->setData(Qt::UserRole, QVariant((*wells)[i].name.split("\t").last()));
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable); // set checkable flag
        if ((*wells)[i].active) {
            item->setCheckState(Qt::Checked);
        } else {
            item->setCheckState(Qt::Unchecked);
        }

        ui->wells->addItem(item);
    }
}

/**
 * @brief WellViewer::reset
 *
 * Reset function to restore window state
 */
void WellViewer::reset() {
    // reset ui and internal lists
    while(mats.size()>logsIndexes.size()) {
        mats.remove(0);
    }
    setUpdatesEnabled(false);
    for (int i = 0; i < logsViews.size(); i++) {
        MuratCanvas2dLogView* view = logsViews[i];
        synchroniser.remove(view);
        viewLayout->removeWidget(view);
        viewLayout->setColumnStretch(i, 0);
        view->deleteLater();

        LogStats* logStats = logsStats[i];
        viewLayout->removeWidget(logStats);
        logStats->deleteLater();
    }
    QLayout* tmpLayout = viewLayout;
    delete tmpLayout;
    viewLayout = new QGridLayout;

    logsViews.clear();
    logsStats.clear();

    setUpdatesEnabled(true);
    updateGeometry();

    // reset selection state
    ui->wells->clearSelection();
    currentWellIndex = -1;
}

void WellViewer::wellChange(QListWidgetItem* item) {
    int i;
    for (i=0; i<wells->count(); i++) {
        if (item->data(Qt::UserRole).toString() == (*wells)[i].name) {
            break;
        }
    }

    if (i<wells->size()) {
        //(wells*)[i].active = item->checkState() == Qt::Checked;
        toggleWell((*wells)[i], item->checkState() == Qt::Checked);
    }
}


void WellViewer::selectLogs(QVector<unsigned int> logsIndexes, QVector<LogParameter> logParameters) {
    this->logsIndexes = logsIndexes;
    this->logsParameters = logParameters;

    while(mats.size()<logsIndexes.size()) {
        mats.append(nullptr);
    }
    while(mats.size()>logsIndexes.size()) {
        mats.remove(0);
    }

    setUpdatesEnabled(false);
    for (int i = 0; i < logsViews.size(); i++) {
        MuratCanvas2dLogView* view = logsViews[i];
        synchroniser.remove(view);
        viewLayout->removeWidget(view);
        viewLayout->setColumnStretch(i, 0);
        view->deleteLater();

        LogStats* logStats = logsStats[i];
        viewLayout->removeWidget(logStats);
        logStats->deleteLater();
    }
    //QGridLayout* tmpLayout = viewLayout;
    holder->removeItem(viewLayout);
    delete viewLayout;
    viewLayout = new QGridLayout;

    holder->addLayout(viewLayout);
    logsViews.clear();
    logsStats.clear();


    for (int i = 0; i < logsIndexes.size(); i++) {
        MuratCanvas2dLogView* view = new MuratCanvas2dLogView();
        view->initScene();
        viewLayout->addWidget(view, 0, i);
        viewLayout->setColumnStretch(i, 1);
        logsViews.append(view);
        view->setCurveOffset(0);
        view->setCurveSize(200);
        synchroniser.addCanvas2d(view);
        logsStats.append(new LogStats);
        viewLayout->addWidget(logsStats.last(), 1, i);
    }
    setUpdatesEnabled(true);
    updateGeometry();

    for (int j=0; j<logsViews.size(); j++) {

        connect(logsViews[j], &MuratCanvas2dLogView::linesChanged, this, [this, j](const std::vector<double> lines) {
            linesChanged(j, lines);
        });
    }

    wellSelectedChanged();
}

void WellViewer::wellSelectedChanged() {
	currentWellIndex = -1;
    if (ui->wells->selectedItems().size()!=1) {
        return;
    }
    QString name = ui->wells->selectedItems()[0]->data(Qt::UserRole).toString();
    int k = 0;

    while ((*wells).size()>k && QString::compare((*wells)[k].name, name)!=0) {
        k++;
    }

    if ((*wells).size()==k) {
        return;
    }
    currentWellIndex = k;
    // Logs
    QImage img(1, (*wells)[k].samples.size(), QImage::Format_Grayscale8);

    for (int j=0; j<logsViews.size(); j++) {
        this->mats[j].reset(new Matrix2DLine<double>(1, (*wells)[k].samples.size()));
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::min();
        double sum = 0;
        double sum2 = 0;

        for (int i=0; i<(*wells)[k].samples.size(); i++) {
            double val = (*wells)[k].samples[i].logVals[logsIndexes[j]];
            if (logPreprocessing.size()==(*wells)[0].samples[0].logVals.size()) {
                if (logPreprocessing[logsIndexes[j]] == LogLn) {
                    if (val > 0) {
                        val = std::log(val);
                    } else {
                        val = std::numeric_limits<double>::lowest();
                    }
                }
            }
            this->mats[j]->set(val,0, i);
            if (val < min && val != std::numeric_limits<double>::lowest()) {
                min = val;
            }
            if (val > max) {
                max = val;
            }
            sum += val;
            sum2 += val*val;
        }

        double mean = sum / (*wells)[k].samples.size();
        double std = std::sqrt((sum2 - sum*sum /(*wells)[k].samples.size() ) / (*wells)[k].samples.size());


        logsViews[j]->setImage(img, (Matrix2DInterface*) this->mats[j].get());

        logsViews[j]->setMaxMat(logsParameters[logsIndexes[j]].InputMax);
        std::vector<double> lines;
        lines.resize((*wells)[k].ranges.size() * 2);
        for (int minMaxIdx=0; minMaxIdx<(*wells)[k].ranges.size(); minMaxIdx++) {
            lines[minMaxIdx*2] = (*wells)[k].ranges[minMaxIdx].min;
            lines[minMaxIdx*2+1] = (*wells)[k].ranges[minMaxIdx].max;
        }
        logsViews[j]->setLines(lines);


        logsViews[j]->setMinMat(logsParameters[logsIndexes[j]].InputMin);
        logsStats[j]->setMax(max);
        logsStats[j]->setMin(min);
        logsStats[j]->setMean(mean);
        logsStats[j]->setStd(std);

        //qDebug() << "Indexes" << (*wells)[k].minIndex << (*wells)[k].maxIndex;

    }

    // Seismic
    double n = 0;
    double sumT = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    double sum;
    double sum2;
    double stdMin = std::numeric_limits<double>::max();

    for(int i=0; i<(*wells)[k].samples.size(); i++) {
        QVector<float>& sample = (*wells)[k].samples[i].seismicVals;
        sum = 0;
        sum2 = 0;
        for (int j=0; j<sample.size(); j++) {
            if (sample[j]<min) {
                min = sample[j];
            }
            if (sample[j]>max) {
                max = sample[j];
            }
            sum += sample[j];
            sum2 += sample[j]*sample[j];
        }
        double std = std::sqrt((sum2 - sum*sum / sample.size()) / sample.size());
        if (stdMin>std) {
            stdMin = std;
        }
        sumT += sum;
        n += sample.size();
    }
    ui->seismicMean->setText(QString::number(sumT/n));
    ui->seismicMin->setText(QString::number(min));
    ui->seismicMax->setText(QString::number(max));
    ui->seismicStd->setText(QString::number(stdMin));
}

bool WellViewer::setLogPreprocessing(QVector<int> array) {

    bool output = wells!=nullptr && (*wells).size()>0 && (*wells)[0].samples[0].logVals.size() == array.size();
    if (output) {
        logPreprocessing = array;
    }
    return output;
}

void WellViewer::linesChanged(int viewIdx, const std::vector<double>& posLines) {
    if (ui->wells->selectedItems().size()!=1) {
        return;
    }
    QString name = ui->wells->selectedItems()[0]->data(Qt::UserRole).toString();
    int k = 0;

    while ((*wells).size()>k && QString::compare((*wells)[k].name, name)!=0) {
        k++;
    }

    if ((*wells).size()==k) {
        return;
    }

    blockSignals(true);
    for (MuratCanvas2dLogView* e : logsViews) {
        if (e != logsViews[viewIdx]) {
            e->setLines(posLines);
        }
    }

    std::vector<double> sortLines = posLines;
    std::sort(sortLines.begin(), sortLines.end());

    // should be even
    (*wells)[k].ranges.resize(sortLines.size()/2);
    for (int minMaxIdx = 0; minMaxIdx<(*wells)[k].ranges.size(); minMaxIdx++) {
        (*wells)[k].ranges[minMaxIdx].min = sortLines[minMaxIdx*2];
        (*wells)[k].ranges[minMaxIdx].max = sortLines[minMaxIdx*2+1];
    }
    blockSignals(false);
    emit this->minMaxIndexChanged(k);
}

void WellViewer::addNewRange() {
	if (logsViews.size()==0 || currentWellIndex<0 || currentWellIndex>=wells->size()) {
	    return;
	}
    std::vector<double> lines = logsViews[0]->getLines();

    if (lines.size()>0) {
        // sort
        std::sort(lines.begin(), lines.end());

        // try to add at the bottom
        int newMax = (*wells)[currentWellIndex].samples.size() - 1;
        int newMin = std::round(lines[lines.size()-1]) + 1;
        if (newMin<newMax) {
            // reduce interval to add to make it easier to grab
            newMin = (newMax - newMin) / 2 + newMin;
        } else {
            // try top
            newMin = 0;
            newMax = std::round(lines[0])-1;
            if (newMin<newMax) {
                // reduce interval to add to make it easier to grab
                newMax = newMax / 2 + 1;
            } else {
                // no space at the top and the bottom, add it at the very bottom
                newMax = (*wells)[currentWellIndex].samples.size() - 1;
                newMin = newMax;
            }
        }
        lines.resize(lines.size()+2);
        lines[lines.size()-2] = newMin;
        lines[lines.size()-1] = newMax;
    } else {
        lines.resize(2);
        lines[0] = 0;
        lines[1] = (*wells)[currentWellIndex].samples.size() - 1;
    }

    blockSignals(true);
    for (int viewIdx=0; viewIdx<logsViews.size(); viewIdx++) {
        logsViews[viewIdx]->setLines(lines);
    }
    blockSignals(false);
}

void WellViewer::removeRange() {
    if (logsViews.size()==0 || currentWellIndex<0 || currentWellIndex>=wells->size()) {
        return;
    }

    std::vector<double> lines = logsViews[0]->getLines();

    if (lines.size()>3) {
        // remove the last two
        std::sort(lines.begin(), lines.end());
        std::vector<double>::iterator it = lines.begin();
        std::advance(it, lines.size()-2);
        lines.erase(it, lines.end());

        blockSignals(true);
        for (int viewIdx=0; viewIdx<logsViews.size(); viewIdx++) {
            logsViews[viewIdx]->setLines(lines);
        }
        blockSignals(false);
    }
}

// return first index of plateau
// plateau minimum size is 2
std::pair<bool, int> WellViewer::searchBeginPlateau(LogSample* tab, int size, const std::vector<int>& logIndexSearch, int plateauMinimumSize) {
	if (size<=0 || plateauMinimumSize<2) {
		return std::pair<bool, int>(false, 0);
	}

	bool plateauBeginFound = false;
	int idx = 0;
	int isSameCount = 0;
	while (!plateauBeginFound && idx<size-1) {
		bool isSame = false;
		int searchIdx = 0;
		while (!isSame && searchIdx<logIndexSearch.size()) {
			// qFuzzyIsNull over qFuzzyCompare because qFuzzyIsNull use an absolute threshold and qFuzzyCompare use a relative threshold
			isSame = qFuzzyIsNull(tab[idx].logVals[logIndexSearch[searchIdx]] -
					tab[idx+1].logVals[logIndexSearch[searchIdx]]);
			if (!isSame) {
				searchIdx++;
			}
		}

		if (isSame && isSameCount==0) {
			isSameCount = 2;
		} else if (isSame) {
			isSameCount++;
		} else {
			isSameCount = 0;
		}

		plateauBeginFound = isSameCount==plateauMinimumSize;
		if (!plateauBeginFound) {
			idx++;
		}
	}
	if (plateauBeginFound) {
		// to return the first index of the plateau
		// idx = plateau begin + (plateauMinimumSize - 2)
		// ex : plateauMinimumSize = 2 then plateauBeginIdx = idx
		int plateauBeginIdx = idx - plateauMinimumSize + 2;

		if (plateauBeginIdx<0) {
			// safety measure
			qDebug() << "WellViewer::searchBeginPlateau ERROR invalid index";
			plateauBeginIdx = 0;
		}
		idx = plateauBeginIdx;
	}
	return std::pair<bool, int>(plateauBeginFound, idx);
}

// return first index of plateau
// plateau minimum size is 2
std::pair<bool, int> WellViewer::searchBeginPlateau(const float* tab, int size, int plateauMinimumSize) {
	if (size<=0 || plateauMinimumSize<2) {
		return std::pair<bool, int>(false, 0);
	}

	bool plateauBeginFound = false;
	int idx = 0;
	int isSameCount = 0;
	while (!plateauBeginFound && idx<size-1) {
		bool isSame = false;
		// qFuzzyIsNull over qFuzzyCompare because qFuzzyIsNull use an absolute threshold and qFuzzyCompare use a relative threshold
		isSame = qFuzzyIsNull(tab[idx] - tab[idx+1]);

		if (isSame && isSameCount==0) {
			isSameCount = 2;
		} else if (isSame) {
			isSameCount++;
		} else {
			isSameCount = 0;
		}

		plateauBeginFound = isSameCount==plateauMinimumSize;
		if (!plateauBeginFound) {
			idx++;
		}
	}
	if (plateauBeginFound) {
		// to return the first index of the plateau
		// idx = plateau begin + (plateauMinimumSize - 2)
		// ex : plateauMinimumSize = 2 then plateauBeginIdx = idx
		int plateauBeginIdx = idx - plateauMinimumSize + 2;

		if (plateauBeginIdx<0) {
			// safety measure
			qDebug() << "WellViewer::searchBeginPlateau ERROR invalid index";
			plateauBeginIdx = 0;
		}
		idx = plateauBeginIdx;
	}
	return std::pair<bool, int>(plateauBeginFound, idx);
}

// return last index of plateau
std::pair<bool, int> WellViewer::searchEndPlateau(LogSample* tab, int size, const std::vector<int>& logIndexSearch) {
	if (size<=0) {
		return std::pair<bool, int>(false, 0);
	}

	bool plateauEndFound = false;
	int idx = 0;
	while (!plateauEndFound && idx<size-1) {
		bool isSame = false;
		int searchIdx = 0;
		while (!isSame && searchIdx<logIndexSearch.size()) {
			// qFuzzyIsNull over qFuzzyCompare because qFuzzyIsNull use an absolute threshold and qFuzzyCompare use a relative threshold
			isSame = qFuzzyIsNull(tab[idx].logVals[logIndexSearch[searchIdx]] -
					tab[idx+1].logVals[logIndexSearch[searchIdx]]);
			if (!isSame) {
				searchIdx++;
			}
		}
		plateauEndFound = !isSame;
		if (!plateauEndFound) {
			idx++;
		}
	}
	return std::pair<bool, int>(plateauEndFound, idx);
}

void WellViewer::changeRangeToRemoveWellPlateaus(Well& well, const std::vector<int>& logIndexSearch) {
	if (well.samples.size()==0) {
		return;
	}

	std::vector<int> realLogIndexSearch = logIndexSearch;
	if (realLogIndexSearch.size()==0) {
		realLogIndexSearch.resize(well.samples[0].logVals.size());
		for (int i=0; i<well.samples[0].logVals.size(); i++) {
			realLogIndexSearch[i] = i;
		}
	} else {
		// clean bad indexes
		for (int i=realLogIndexSearch.size()-1; i>=0; i--) {
			if (realLogIndexSearch[i]<0 || realLogIndexSearch[i]>=well.samples[0].logVals.size()) {
				std::vector<int>::iterator it = realLogIndexSearch.begin();
				std::advance(it, i);
				realLogIndexSearch.erase(it);
			}
		}
		if (realLogIndexSearch.size()==0) {
			// all indexes were cleared, choose to exit rather than to use all indexes
			return;
		}
	}

	QVector<IntRange> newRanges;
	for (int i=0; i<well.ranges.size(); i++) {
		int begin = well.ranges[i].min;
		int end = well.ranges[i].max;

		int currentBegin = begin;

		if (begin<end) {
			while (currentBegin<end) {
				int idx = currentBegin;
				// search begin plateau
				std::pair<bool, int> searchResult = searchBeginPlateau(well.samples.data()+idx, end-idx+1, realLogIndexSearch, 5);

				if (searchResult.first) {
					// add space before plateau as a range if it is valid
					idx = currentBegin + searchResult.second;
					int foundEnd = idx - 1; // do not include plateau begin in the range
					if (currentBegin<foundEnd) {
						IntRange range;
						range.min = currentBegin;
						range.max = foundEnd;
						newRanges.push_back(range);
					} // else the range is too thin

					// search end plateau
					std::pair<bool, int> searchEndResult = searchEndPlateau(well.samples.data()+idx, end-idx+1, realLogIndexSearch);
					if (searchEndResult.first) {
						currentBegin = idx + searchEndResult.second + 1; // avoid last index of plateau
					} else {
						currentBegin = end;
					}
				} else {
					IntRange lastRange;
					lastRange.min = currentBegin;
					lastRange.max = end;
					newRanges.push_back(lastRange);
					currentBegin = end + 1; // to end while loop
				}

			}
		} else  {
			// nothing to do keep old range
			newRanges.push_back(well.ranges[i]);
		}
	}

	if (newRanges.size()!=0) {
		well.ranges = newRanges;
	} else {
		// keep old ranges
		well.active = false;
	}

	well.ranges = newRanges;
}

void WellViewer::removeWellsWithConstantSeismic(Well& well, int nVolumes, int signalHalfWindowSize) {
    if (!well.active) {
        // skip hidden wells
        return;
    }

    QVector<LogSample>& samples = well.samples;

    if (samples.size()==0 || samples[0].seismicVals.size()==0) {
        return;
    }

    QVector<IntRange> oldRanges = well.ranges;
    QVector<IntRange> newRanges;
    for (int rangeIdx=0; rangeIdx<oldRanges.size(); rangeIdx++) {
        int lastMin = oldRanges[rangeIdx].min;
        for (int sampleIdx=oldRanges[rangeIdx].min; sampleIdx<=oldRanges[rangeIdx].max; sampleIdx++) {
            bool sampleValid = isSeismicWithPlateau(samples[sampleIdx].seismicVals, nVolumes, 1+2*signalHalfWindowSize);
            if (!sampleValid) {
                IntRange range;
                range.min = lastMin;
                range.max = sampleIdx-1;
                if (range.min<=range.max) {
                    newRanges.append(range);
                }
                lastMin = sampleIdx+1;
            }
        }
        if (lastMin<=oldRanges[rangeIdx].max) {
            IntRange range;
            range.min = lastMin;
            range.max = oldRanges[rangeIdx].max;
            newRanges.append(range);
        }
    }

    if (newRanges.size()==0) {
        well.active = false;
    }

    well.ranges = newRanges;
}

bool WellViewer::isSeismicWithPlateau(const QVector<float>& seismicValues, int nVolume, int seismicWindowSize) {
    if (nVolume*seismicWindowSize!=seismicValues.size()) {
        // invalid sizes
        return false;
    }

    bool seismicValid = true;

    int volumeIdx = 0;
    while (seismicValid && volumeIdx<nVolume) {
        std::pair<bool, int> plateauPostion = searchBeginPlateau(seismicValues.data()+seismicWindowSize*volumeIdx, seismicWindowSize, 3);
        seismicValid = !plateauPostion.first;

        volumeIdx++;
    }

    return seismicValid;
}
