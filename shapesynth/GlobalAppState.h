#pragma once

#include "stdafx.h"

#include <vector>
#include <string>
#include <list>

#define X_GLOBAL_APP_STATE_PATCH_FIELDS \
	X(unsigned int, s_numLevels) \
	X(unsigned int, s_numItersPerLevel) \
	X(float, s_truncationDistance) \
	X(std::string, s_baseDataDir) \
	X(std::string, s_inputSdfFile) \
	X(std::string, s_completedDfFile) \
	X(std::vector<std::string>, s_neighborFiles) \
	X(unsigned int, s_searchMaxK) \
	X(float, s_searchEps) \
	X(int, s_radiusFine) \
	X(int, s_radiusCoarse) \
	X(bool, s_useCoherence) \
	X(float, s_coherenceKappa) \
	X(unsigned int, s_featureDim) \
	X(std::string, s_outputDir) \
	X(bool, s_verbose) \
	X(unsigned int, s_startLevel) 


#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif

#define checkSizeArray(a, d)( (((sizeof a)/(sizeof a[0])) >= d))

class GlobalAppState
{
public:

#define X(type, name) type name;
	X_GLOBAL_APP_STATE_PATCH_FIELDS
#undef X

		//! sets the parameter file and reads
	void readMembers(const ParameterFile& parameterFile) {
		m_ParameterFile = parameterFile;
		readMembers();
	}

	//! reads all the members from the given parameter file (could be called for reloading)
	void readMembers() {
#define X(type, name) \
	if (!m_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
		X_GLOBAL_APP_STATE_PATCH_FIELDS
#undef X
 

		m_bIsInitialized = true;
	}

	void print() const {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
		X_GLOBAL_APP_STATE_PATCH_FIELDS
#undef X
	}

	static GlobalAppState& getInstance() {
		static GlobalAppState s;
		return s;
	}
	static GlobalAppState& get() {
		return getInstance();
	}


	//! constructor
	GlobalAppState() {
		m_bIsInitialized = false;
	}

	//! destructor
	~GlobalAppState() {
	}

	Timer	s_Timer;

private:
	bool			m_bIsInitialized;
	ParameterFile	m_ParameterFile;
};
