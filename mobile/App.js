import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { Video, ResizeMode } from 'expo-av';

const API_BASE_URL = 'http://192.168.1.216:8000'; // replace with your backend IP

export default function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [loading, setLoading] = useState(false);

  const [analysisResult, setAnalysisResult] = useState(null);
  const [feedback, setFeedback] = useState([]);
  const [repFeedback, setRepFeedback] = useState([]);
  const [setSummary, setSetSummary] = useState(null);

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setFeedback([]);
    setRepFeedback([]);
    setSetSummary(null);
  };

  const normalizePickedFile = (asset) => {
    if (!asset) return null;

    return {
      uri: asset.uri,
      name: asset.name || 'upload.mp4',
      mimeType: asset.mimeType || 'video/mp4',
      size: asset.size || null,
    };
  };

  const pickFromFiles = async () => {
    try {
      resetAnalysis();

      const result = await DocumentPicker.getDocumentAsync({
        type: 'video/*',
        copyToCacheDirectory: true,
        multiple: false,
      });

      if (result.canceled) return;

      const asset = result.assets?.[0];
      if (!asset) return;

      setSelectedVideo(normalizePickedFile(asset));
    } catch (error) {
      console.error('pickFromFiles error:', error);
      Alert.alert('Error', 'Could not open file picker.');
    }
  };

  const pickFromLibrary = async () => {
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission needed', 'Please allow photo library access.');
        return;
      }

      resetAnalysis();

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });

      if (result.canceled) return;

      const asset = result.assets?.[0];
      if (!asset) return;

      setSelectedVideo({
        uri: asset.uri,
        name: asset.fileName || 'library-video.mp4',
        mimeType: asset.mimeType || 'video/mp4',
        size: asset.fileSize || null,
      });
    } catch (error) {
      console.error('pickFromLibrary error:', error);
      Alert.alert('Error', 'Could not pick a video from library.');
    }
  };

  const recordVideo = async () => {
    try {
      const permission = await ImagePicker.requestCameraPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission needed', 'Please allow camera access.');
        return;
      }

      resetAnalysis();

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
        videoMaxDuration: 60,
      });

      if (result.canceled) return;

      const asset = result.assets?.[0];
      if (!asset) return;

      setSelectedVideo({
        uri: asset.uri,
        name: asset.fileName || 'camera-video.mp4',
        mimeType: asset.mimeType || 'video/mp4',
        size: asset.fileSize || null,
      });
    } catch (error) {
      console.error('recordVideo error:', error);
      Alert.alert('Error', 'Could not record video.');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedVideo?.uri) {
      Alert.alert('No video selected', 'Please choose or record a video first.');
      return;
    }

    try {
      setLoading(true);
      resetAnalysis();

      const formData = new FormData();
      formData.append('file', {
        uri: selectedVideo.uri,
        name: selectedVideo.name || 'upload.mp4',
        type: selectedVideo.mimeType || 'video/mp4',
      });

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
        },
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        console.log('Backend error:', text);
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      console.log('FULL BACKEND RESPONSE:', JSON.stringify(data, null, 2));

      setAnalysisResult(data);
      setFeedback(Array.isArray(data.feedback) ? data.feedback : []);
      setRepFeedback(Array.isArray(data.rep_feedback) ? data.rep_feedback : []);

      let summary = data.set_summary ?? null;

      if (typeof summary === 'string') {
        try {
          summary = JSON.parse(summary);
        } catch (e) {
          console.log('Could not parse set_summary:', summary);
        }
      }

      setSetSummary(summary);
    } catch (error) {
      console.error('Analyze error:', error);
      Alert.alert('Analysis failed', error.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const renderPredictionCard = () => {
    if (!analysisResult) return null;

    const confidence =
      analysisResult.confidence != null
        ? `${(analysisResult.confidence * 100).toFixed(1)}% confidence`
        : 'No confidence available';

    return (
      <View style={styles.predictionCard}>
        <Text style={styles.predictionLabel}>Prediction</Text>
        <Text style={styles.predictionTitle}>
          {analysisResult.exercise_label || 'Unknown'}
        </Text>
        <Text style={styles.predictionConfidence}>{confidence}</Text>
      </View>
    );
  };

  const renderSetSummary = () => {
    if (!setSummary) return null;

    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Set Summary</Text>

        <Text style={styles.summaryRow}>
          Detected Reps: {setSummary.detected_reps ?? 0}
        </Text>
        <Text style={styles.summaryRow}>
          Avg Score: {setSummary.avg_rep_score != null ? setSummary.avg_rep_score : '—'}
        </Text>
        <Text style={styles.summaryRow}>
          Best Rep: {setSummary.best_rep != null ? setSummary.best_rep : '—'}
        </Text>
        <Text style={styles.summaryRow}>
          Worst Rep: {setSummary.worst_rep != null ? setSummary.worst_rep : '—'}
        </Text>
        <Text style={styles.summaryRow}>
          Trend: {setSummary.trend ?? 'No summary available.'}
        </Text>
      </View>
    );
  };

  const renderOverallFeedback = () => {
    if (!feedback.length) return null;

    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Overall Feedback</Text>
        {feedback.map((item, index) => (
          <Text key={index} style={styles.bulletText}>
            • {item}
          </Text>
        ))}
      </View>
    );
  };

  const renderRepFeedback = () => {
    if (!repFeedback.length) return null;

    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Per-Rep Breakdown</Text>

        {repFeedback.map((rep, index) => (
          <View key={index} style={styles.repBox}>
            <Text style={styles.repTitle}>
              Rep {rep.rep ?? index + 1}
              {rep.grade ? ` — ${rep.grade}` : ''}
            </Text>

            <Text style={styles.repText}>
              Score: {rep.score != null ? rep.score : '—'}
            </Text>

            {rep.breakdown && (
              <>
                <Text style={styles.repText}>Depth: {rep.breakdown.depth ?? '—'}</Text>
                <Text style={styles.repText}>Knees: {rep.breakdown.knees ?? '—'}</Text>
                <Text style={styles.repText}>Torso: {rep.breakdown.torso ?? '—'}</Text>
              </>
            )}

            {Array.isArray(rep.issues) && rep.issues.length > 0 && (
              <>
                <Text style={styles.repSubTitle}>Issues</Text>
                {rep.issues.map((issue, issueIndex) => (
                  <Text key={issueIndex} style={styles.bulletText}>
                    • {issue}
                  </Text>
                ))}
              </>
            )}

            {Array.isArray(rep.feedback) && rep.feedback.length > 0 && (
              <>
                <Text style={styles.repSubTitle}>Feedback</Text>
                {rep.feedback.map((fb, fbIndex) => (
                  <Text key={fbIndex} style={styles.bulletText}>
                    • {fb}
                  </Text>
                ))}
              </>
            )}
          </View>
        ))}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.header}>FormCheck AI</Text>
        <Text style={styles.subHeader}>
          Pick a video from Files, Photos, or record one with your camera.
        </Text>

        <View style={styles.buttonColumn}>
          <TouchableOpacity style={styles.actionButton} onPress={pickFromFiles}>
            <Text style={styles.buttonText}>Pick From Files</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionButton} onPress={pickFromLibrary}>
            <Text style={styles.buttonText}>Pick From Photos</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionButton} onPress={recordVideo}>
            <Text style={styles.buttonText}>Record Video</Text>
          </TouchableOpacity>
        </View>

        {selectedVideo && (
          <View style={styles.videoCard}>
            <Text style={styles.cardTitle}>Selected Video</Text>
            <Text style={styles.fileName}>{selectedVideo.name || 'Video selected'}</Text>

            <Video
              source={{ uri: selectedVideo.uri }}
              style={styles.video}
              useNativeControls
              resizeMode={ResizeMode.CONTAIN}
              shouldPlay={false}
            />
          </View>
        )}

        <TouchableOpacity
          style={[styles.analyzeButton, loading && styles.buttonDisabled]}
          onPress={handleAnalyze}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </Text>
        </TouchableOpacity>

        {loading && <ActivityIndicator size="large" style={styles.loader} />}

        {renderPredictionCard()}
        {renderSetSummary()}
        {renderOverallFeedback()}
        {renderRepFeedback()}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#031226',
  },
  container: {
    padding: 20,
    paddingBottom: 40,
    backgroundColor: '#031226',
  },
  header: {
    fontSize: 32,
    fontWeight: '800',
    color: '#ffffff',
    marginBottom: 8,
  },
  subHeader: {
    fontSize: 16,
    color: '#a9b4c2',
    marginBottom: 20,
  },
  buttonColumn: {
    marginBottom: 16,
  },
  actionButton: {
    backgroundColor: '#12365d',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  analyzeButton: {
    backgroundColor: '#1e7a46',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
  },
  loader: {
    marginBottom: 20,
  },
  videoCard: {
    backgroundColor: '#081a33',
    borderWidth: 1,
    borderColor: '#1a2c4d',
    borderRadius: 18,
    padding: 16,
    marginBottom: 20,
  },
  fileName: {
    color: '#a9b4c2',
    fontSize: 14,
    marginBottom: 10,
  },
  video: {
    width: '100%',
    height: 240,
    borderRadius: 12,
    backgroundColor: '#000',
    marginTop: 8,
  },
  predictionCard: {
    backgroundColor: '#072d1e',
    borderWidth: 1,
    borderColor: '#1e7a46',
    borderRadius: 20,
    padding: 20,
    marginBottom: 18,
  },
  predictionLabel: {
    color: '#8ee6b5',
    fontSize: 14,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 10,
  },
  predictionTitle: {
    color: '#ffffff',
    fontSize: 36,
    fontWeight: '800',
    marginBottom: 10,
  },
  predictionConfidence: {
    color: '#d8e1ea',
    fontSize: 18,
  },
  card: {
    backgroundColor: '#081a33',
    borderWidth: 1,
    borderColor: '#1a2c4d',
    borderRadius: 18,
    padding: 18,
    marginBottom: 18,
  },
  cardTitle: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 12,
  },
  summaryRow: {
    color: '#e6edf5',
    fontSize: 16,
    marginBottom: 10,
    lineHeight: 24,
  },
  bulletText: {
    color: '#d8e1ea',
    fontSize: 16,
    lineHeight: 28,
    marginBottom: 6,
  },
  repBox: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#1a2c4d',
  },
  repTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  repSubTitle: {
    color: '#8ee6b5',
    fontSize: 15,
    fontWeight: '700',
    marginTop: 10,
    marginBottom: 6,
  },
  repText: {
    color: '#d8e1ea',
    fontSize: 15,
    marginBottom: 6,
  },
});